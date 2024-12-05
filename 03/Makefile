all: test bench bench_async bench_collect

build:
	mpic++ -o task1 task1.cpp

test: build
	mpirun -np 24 ./task1

bench:
	mpic++ -o mpi_bench mpi_bench.cpp
	bash -c 'for N in 10000 25000 50000; do \
        for P in 1 2 4 8 16 24; do \
          echo "Running with N=$$N and P=$$P"; \
          mpirun -np $$P ./mpi_bench $$N >> results.txt; \
        done; \
      done'

bench_async:
	mpic++ -o mpi_bench_async mpi_bench_async.cpp
	bash -c 'for N in 10000 25000 50000; do \
        for P in 1 2 4 8 16 24; do \
          echo "Running with N=$$N and P=$$P"; \
          mpirun -np $$P ./mpi_bench_async $$N >> results_async.txt; \
        done; \
      done'

bench_collect:
	mpic++ -o mpi_bench_collect mpi_bench_collect.cpp
	mpirun -np 24 ./mpi_bench_collect
	
clean:
	echo "------- Clean all -------"
	rm -rf task1 mpi_bench results.txt mpi_bench_async results_async.txt mpi_bench_collect output.txt
