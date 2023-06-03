all: 
	gcc  -o CPU  MatrixMult.c
	nvcc -o Naive	MatrixMultNaive.cu
	nvcc -o Tiled  MatrixMultTiled.cu
run: all
	./CPU 5
	./Naive 5
	./Tiled 5
	./CPU 10
	./Naive 10
	./Tiled 10
	./CPU 20
	./Naive 20
	./Tiled 20
	./CPU 50
	./Naive 50
	./Tiled 50
	./CPU 1021
	./Naive 1021
	./Tiled 1021
	
	
	
clean:
	rm -f CPU 
	rm -f Naive 
	rm -f Tiled
	rm -f Product.out
	rm -f Matrix_Calulations_of_Size_5.dat
	rm -f Matrix_Calulations_of_Size_10.dat
	rm -f Matrix_Calulations_of_Size_20.dat
	rm -f Matrix_Calulations_of_Size_50.dat
	rm -f Matrix_Calulations_of_Size_1021.dat
	rm -f Matrix_Calulations_of_Size_2015.dat
	rm -f CPU.csv
	rm -f Naive.csv
	rm -f Tiled.csv