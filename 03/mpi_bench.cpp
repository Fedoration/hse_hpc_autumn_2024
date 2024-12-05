#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

const double k = 1.0;
const double tau = 2e-10;
const double l = 1.0;
const double T = 0.0001; 
const double u0 = 1.0;

void check_stability(double h) {
    if (tau >= h * h / k) {
        std::cerr << "Условие устойчивости не выполнено!" << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N; 
    if (argc < 2) {
        if (rank == 0) std::cerr << "Укажите количество точек N как аргумент!" << std::endl;
        MPI_Finalize();
        return -1;
    }
    N = std::stoi(argv[1]);
    double h = k / N;

    check_stability(h);

    int points_per_process = N / size;
    int remainder = N % size;

    // Переменные для локальных данных
    int local_size;
    std::vector<double> local_u;
    std::vector<double> new_u;

    // Процесс 0 создает данные и распределяет их
    if (rank == 0) {
        std::vector<double> initial_data(N, u0);

        for (int i = 1; i < size; ++i) {
            int send_start = i * points_per_process + std::min(i, remainder);
            int send_size = (i < remainder) ? points_per_process + 1 : points_per_process;
            MPI_Send(initial_data.data() + send_start, send_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        // Определяем границы и размер локальных данных для процесса 0
        int start = 0;
        int end = points_per_process + (remainder > 0 ? 1 : 0);
        local_size = end - start;

        // Выделяем и заполняем локальный массив для процесса 0
        local_u.resize(local_size);
        std::copy(initial_data.begin(), initial_data.begin() + local_size, local_u.begin());
    } else {
        // Вычисляем размер данных для текущего процесса
        int start = rank * points_per_process + std::min(rank, remainder);
        int end = start + points_per_process + (rank < remainder ? 1 : 0);
        local_size = end - start;

        // Выделяем память для локальных данных
        local_u.resize(local_size);
        new_u.resize(local_size);

        // Получаем данные от процесса 0
        MPI_Recv(local_u.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    new_u.resize(local_size, 0.0);

    int time_steps = static_cast<int>(T / tau);

    double start_time = MPI_Wtime();

    for (int step = 0; step < time_steps; ++step) {
        double left_boundary = 0.0, right_boundary = 0.0;

        if (rank > 0) {
            MPI_Send(&local_u[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&left_boundary, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(&local_u[local_size - 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&right_boundary, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

    // Сбор результатов на процесс 0
    if (rank == 0) {
        std::vector<double> global_u(N);
        std::copy(local_u.begin(), local_u.end(), global_u.begin());
        for (int i = 1; i < size; ++i) {
            int recv_start = i * points_per_process + std::min(i, remainder);
            int recv_size = (i < remainder) ? points_per_process + 1 : points_per_process;
            MPI_Recv(global_u.data() + recv_start, recv_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;
        std::cout << "N: " << N << ", Processes: " << size << ", Time: " << elapsed_time << " seconds\n";

    } else {
        MPI_Send(local_u.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
