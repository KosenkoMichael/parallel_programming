#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

std::vector<std::vector<int> > generateMatrix(int rows, int cols, int minVal, int maxVal) {
    if (minVal > maxVal) std::swap(minVal, maxVal);

    std::vector<std::vector<int> > matrix(rows, std::vector<int>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(minVal, maxVal);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = dist(gen);

    return matrix;
}

void writeMatrixToFile(const std::string& filename, const std::vector<std::vector<int> >& matrix) {
    std::ofstream out(filename.c_str());
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j)
            out << matrix[i][j] << " ";
        out << "\n";
    }
}

bool directoryExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return false;
    return (info.st_mode & S_IFDIR);
}

void createDirectory(const std::string& path) {
    if (!directoryExists(path)) {
        mkdir(path.c_str(), 0755);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0, minVal = 0, maxVal = 10;
    std::string subpath = "";

    if (rank == 0) {
        if (argc < 5) {
            std::cerr << "Usage: mpirun -np <procs> ./program <N> <minVal> <maxVal> <subpath>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N = std::atoi(argv[1]);
        minVal = std::atoi(argv[2]);
        maxVal = std::atoi(argv[3]);
        subpath = argv[4];
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&minVal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rowsA = N, colsA = N;
    int rowsB = N, colsB = N;

    std::vector<std::vector<int> > A, B;
    std::vector<int> A_flat, B_flat(rowsB * colsB);

    if (rank == 0) {
        A = generateMatrix(rowsA, colsA, minVal, maxVal);
        B = generateMatrix(rowsB, colsB, minVal, maxVal);

        A_flat.resize(rowsA * colsA);
        for (int i = 0; i < rowsA; ++i)
            for (int j = 0; j < colsA; ++j)
                A_flat[i * colsA + j] = A[i][j];

        for (int i = 0; i < rowsB; ++i)
            for (int j = 0; j < colsB; ++j)
                B_flat[i * colsB + j] = B[i][j];
    }

    MPI_Bcast(&B_flat[0], rowsB * colsB, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = rowsA / size;
    int remainder = rowsA % size;
    int my_rows = local_rows + (rank < remainder ? 1 : 0);

    std::vector<int> local_A(my_rows * colsA);
    std::vector<int> local_C(my_rows * colsB);

    std::vector<int> sendcounts(size), displs(size);
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int rows = local_rows + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * colsA;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Scatterv(
        &A_flat[0], &sendcounts[0], &displs[0], MPI_INT,
        &local_A[0], my_rows * colsA, MPI_INT,
        0, MPI_COMM_WORLD
    );

    auto local_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < colsB; ++j) {
            int sum = 0;
            for (int k = 0; k < colsA; ++k)
                sum += local_A[i * colsA + k] * B_flat[k * colsB + j];
            local_C[i * colsB + j] = sum;
        }
    }
    auto local_end = std::chrono::high_resolution_clock::now();
    long long local_time = std::chrono::duration_cast<std::chrono::milliseconds>(local_end - local_start).count();

    std::vector<int> C_flat;
    std::vector<int> recvcounts(size), recvdispls(size);
    if (rank == 0) {
        C_flat.resize(rowsA * colsB);
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int rows = local_rows + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * colsB;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
    }

    MPI_Gatherv(
        &local_C[0], my_rows * colsB, MPI_INT,
        &C_flat[0], &recvcounts[0], &recvdispls[0], MPI_INT,
        0, MPI_COMM_WORLD
    );

    long long max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<std::vector<int> > C(rowsA, std::vector<int>(colsB));
        for (int i = 0; i < rowsA; ++i)
            for (int j = 0; j < colsB; ++j)
                C[i][j] = C_flat[i * colsB + j];

        std::stringstream ss;
        ss << size;
        std::string folderName = ss.str() + "_" + std::to_string(N) + subpath;
        std::string basePath = "result/" + folderName;
        createDirectory("result");
        createDirectory(basePath);

        writeMatrixToFile(basePath + "/A.txt", A);
        writeMatrixToFile(basePath + "/B.txt", B);
        writeMatrixToFile(basePath + "/C.txt", C);

        std::cout << "Threads: " << size << ", Matrix size: " << N << "x" << N << " " << max_time << " ms" << std::endl;

        std::string statsFile = "stats.csv";
        std::ifstream infile(statsFile.c_str());
        bool writeHeader = !infile.good();
        infile.close();

        std::ofstream stats(statsFile.c_str(), std::ios::app);
        if (writeHeader) {
            stats << "Threads,MatrixSize,MinValue,MaxValue,Time(ms)\n";
        }
        stats << size << "," << N << "," << minVal << "," << maxVal << "," << max_time << "\n";
        stats.close();
    }

    MPI_Finalize();
    return 0;
}
