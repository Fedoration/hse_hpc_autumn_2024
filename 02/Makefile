all: cuda_run openmp_run

cuda_run:
	nvcc -o matrix_mult.out main.cu -lcublas
	srun -n 1 --gpus=1 -A proj_1593 ./matrix_mult.out

openmp_run: 
	nvc -mp=gpu -o openmp_matrix_mul.out openmp_matrix_mul.c
	srun -n 1 --gpus=1 -A proj_1593 ./openmp_matrix_mul.out

clean:
	echo "------- Clean all -------"
	rm -rf *.out
	rm -rf *.o
