#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

const double k = 1.0;  
const double h = 0.02; 
const double tau = 2e-10;
const double l = 1.0;  
const double T = 0.0001;  
const double u0 = 1.0; 

void check_stability() {
    if (tau >= h * h / k) {
        std::cerr << "Условие устойчивости не выполнено!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Функция для вычисления точного решения
std::vector<double> exact_solution(int N, double t) {
    std::vector<double> exact_u(N, 0.0);
    double pi = M_PI;
    for (int i = 0; i < N; i++) {
        double x = i * h;
        for (int m = 0; m < 1000; m++) { // Обрезаем сумму
            double term = std::exp(-k * (2 * m + 1) * (2 * m + 1) * pi * pi * t / (l * l)) / (2 * m + 1);
            exact_u[i] += term * std::sin((2 * m + 1) * pi * x / l);
        }
        exact_u[i] *= 4 * u0 / pi;
    }
    return exact_u;
}

// Основная программа
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    check_stability();

    int N = static_cast<int>(l / h) + 1;
    int points_per_process = N / size;
    int remainder = N % size;

    // Определяем размеры локальных массивов для каждого процесса
    std::vector<int> sendcounts(size); // Количество элементов для каждого процесса
    std::vector<int> displs(size);     // Смещения для каждого процесса

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = points_per_process + (i < remainder ? 1 : 0);
            displs[i] = (i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1]);
        }
    }

    int local_size;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> local_u(local_size);
    std::vector<double> new_u(local_size, 0.0);

    // Распределение начальных значений
    if (rank == 0) {
        std::vector<double> initial_u(N, u0);
        MPI_Scatterv(initial_u.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                     local_u.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     local_u.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int time_steps = static_cast<int>(T / tau);
    for (int step = 0; step < time_steps; ++step) {
        double left_boundary = 0.0, right_boundary = 0.0;

        if (rank > 0) {
            MPI_Sendrecv(&local_u[0], 1, MPI_DOUBLE, rank - 1, 0,
                         &left_boundary, 1, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&local_u[local_size - 1], 1, MPI_DOUBLE, rank + 1, 0,
                         &right_boundary, 1, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = 0; i < local_size; ++i) {
            double left = (i > 0) ? local_u[i - 1] : left_boundary;
            double right = (i < local_size - 1) ? local_u[i + 1] : right_boundary;
            new_u[i] = local_u[i] + k * tau / (h * h) * (right - 2 * local_u[i] + left);
        }

        if (rank == 0) {
            new_u[0] = 0.0;
        } else if (rank == size - 1) {
            new_u[local_size - 1] = 0.0;
        }

        local_u.swap(new_u);
    }

    // Сбор данных обратно на процессе 0
    std::vector<double> global_u(rank == 0 ? N : 0);
    MPI_Gatherv(local_u.data(), local_size, MPI_DOUBLE, global_u.data(),
                sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<double> exact_u = exact_solution(N, T);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Численное решение: ";
        for (int i = 0; i < N; i += static_cast<int>(0.1 / h)) {
            std::cout << global_u[i] << " ";
        }
        std::cout << "\nТочное решение: ";
        for (int i = 0; i < N; i += static_cast<int>(0.1 / h)) {
            std::cout << exact_u[i] << " ";
        }
        std::cout << "\nРазница: ";
        for (int i = 0; i < N; i += static_cast<int>(0.1 / h)) {
            std::cout << std::abs(global_u[i] - exact_u[i]) << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
