$threadCounts = @(1, 2, 4, 8, 12)

foreach ($threads in $threadCounts) {
    for ($size = 100; $size -le 1000; $size += 100) {
        for ($i = 0; $i -le 5; $i++) {
            Write-Host "Running with $threads threads, size $size, iteration $i"
            mpiexec -np $threads .\build\Debug\matrix_mpi.exe $size 0 100 .$i
        }
    }
}

python check.py

python graph_draw.py