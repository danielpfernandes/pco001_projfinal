	
all: datcsv pco001

datcsv:
	clang++ tools/convert_dat_csv.cpp -std=c++1y -o tools/datcsv -Iinclude -O3 -Wall

pco001:
	clang++ samples/pco001_test.cpp -std=c++1y -o pco001_test -Iinclude -O3 -fopenmp -Wall
	
clean:
	rm -f pco001_test tools/datcsv teste.dat *.txt data/pco001/*.test data/pco001/*.training data/pco001/*.referencia

