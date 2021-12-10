	
all: test openmp persistence unsup anomaly datcsv pco001
	

datcsv:
	clang++ tools/convert_dat_csv.cpp -std=c++1y -o tools/datcsv -Iinclude -O3 -Wall

pco001:
	clang++ samples/pco001_test.cpp -std=c++1y -o pco001_test -Iinclude -O3 -fopenmp -Wall
	
clean:
	rm -f pco001_test test test_parallel persistence test_unsup test_anomaly tools/datcsv teste.dat timing.txt training.txt

