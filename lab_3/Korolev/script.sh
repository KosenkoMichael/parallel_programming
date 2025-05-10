#!/bin/bash    
for (( size=100; size<=1000; size+=100 )); do
        for (( i=0; i<=5; i++ )); do
            echo "Running with $threads threads, size $size, iteration $i"
            mpirun -r ssh ./matrix_mpi "$size" 0 100 "_$i"
        done
done
