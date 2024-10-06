CXX = g++
CXXFLAGS = -fopenmp
BLASFLAGS = -lopenblas
MKLROOT = /opt/intel/mkl
MKL_INCLUDES = -I$(MKLROOT)/include
MKL_LIBS_SEQ = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

all: test bench

build_bench:
	$(CXX) gemm_laputin.cpp bench.cpp -o bench.o -fopenmp

build_blas: 
	$(CXX) gemm_laputin.cpp bench_blas.cpp -o bench_blas.o -fopenmp $(BLASFLAGS)

build_mkl:
	$(CXX) gemm_laputin.cpp bench_mkl.cpp -o bench_mkl.o -fopenmp $(MKL_LIBS_SEQ) $(MKL_INCLUDES)

build_custom_parallel: 
	$(CXX) gemm_laputin.cpp bench_custom_parallel.cpp -o bench_custom_parallel.o -fopenmp $(BLASFLAGS) 

test: build_blas build_mkl build_custom_parallel
	OMP_NUM_THREADS=24 ./bench_blas.o
	OMP_NUM_THREADS=24 ./bench_mkl.o
	OMP_NUM_THREADS=24 ./bench_custom_parallel.o

bench: build_bench
	OMP_NUM_THREADS=24 ./bench.o

clean:
	echo "------- Clean all -------"
	rm -rf *.out
	rm -rf *.o
