#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <direct.h>

#define CUDA_CHECK(err) if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
    exit(1); \
}

void createDirectory(const std::string& path) {
    std::string command = "mkdir \"" + path + "\"";
    std::system(command.c_str());
}

std::vector<std::vector<int>> generateMatrix(int rows, int cols, int minVal, int maxVal) {
    if (minVal > maxVal) std::swap(minVal, maxVal);
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(minVal, maxVal);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = dist(gen);

    return matrix;
}

void writeMatrixToFile(const std::string& filename, const std::vector<std::vector<int>>& matrix) {
    std::ofstream out(filename);
    for (const auto& row : matrix) {
        for (int val : row)
            out << val << " ";
        out << "\n";
    }
}

__global__ void matrixMultiplyKernel(int* A, int* B, int* C, int N) {
    extern __shared__ int sharedMem[];

    int* sharedA = sharedMem;
    int* sharedB = &sharedMem[blockDim.x * blockDim.y];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

    for (int t = 0; t < (N + blockDim.x - 1) / blockDim.x; ++t) {
        int indexA = threadIdx.y * blockDim.x + threadIdx.x;
        int indexB = threadIdx.y * blockDim.y + threadIdx.x;

        if (row < N && t * blockDim.x + threadIdx.x < N)
            sharedA[indexA] = A[row * N + t * blockDim.x + threadIdx.x];
        else
            sharedA[indexA] = 0;

        if (col < N && t * blockDim.y + threadIdx.y < N)
            sharedB[indexB] = B[(t * blockDim.y + threadIdx.y) * N + col];
        else
            sharedB[indexB] = 0;

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k)
            sum += sharedA[threadIdx.y * blockDim.x + k] * sharedB[k * blockDim.y + threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: ./cuda_matrix_mul <N> <minVal> <maxVal> <subfolder> <threadsX> <threadsY>\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    int minVal = std::atoi(argv[2]);
    int maxVal = std::atoi(argv[3]);
    std::string subpath = argv[4];
    int threadsX = std::atoi(argv[5]);
    int threadsY = std::atoi(argv[6]);

    if (threadsX <= 0 || threadsY <= 0 || threadsX * threadsY > 1024) {
        std::cerr << "Error: threadsX * threadsY must be > 0 and <= 1024\n";
        return 1;
    }

    auto A = generateMatrix(N, N, minVal, maxVal);
    auto B = generateMatrix(N, N, minVal, maxVal);

    std::vector<int> A_flat(N * N), B_flat(N * N), C_flat(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A_flat[i * N + j] = A[i][j];
            B_flat[i * N + j] = B[i][j];
        }

    int* d_A;
    int* d_B;
    int* d_C;
    size_t size = N * N * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_flat.data(), size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(threadsX, threadsY);
    dim3 blocksPerGrid((N + threadsX - 1) / threadsX, (N + threadsY - 1) / threadsY);

    size_t sharedMemSize = 2 * threadsX * threadsY * sizeof(int);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    long long elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    CUDA_CHECK(cudaMemcpy(C_flat.data(), d_C, size, cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> C(N, std::vector<int>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = C_flat[i * N + j];

    std::string folderName = "result\\cuda_" + std::to_string(N) + subpath;
    createDirectory("result");
    createDirectory(folderName);

    writeMatrixToFile(folderName + "\\A.txt", A);
    writeMatrixToFile(folderName + "\\B.txt", B);
    writeMatrixToFile(folderName + "\\C.txt", C);

    std::cout << "CUDA Matrix Multiplication Completed\n";
    std::cout << "Matrix size: " << N << "x" << N << " Time: " << elapsedTime << " microseconds\n";

    bool isNewFile = false;
    std::ifstream test("stats.csv");
    if (!test.good() || test.peek() == std::ifstream::traits_type::eof())
        isNewFile = true;
    test.close();

    std::ofstream stats("stats.csv", std::ios::app);
    if (isNewFile)
        stats << "ThreadsX,ThreadsY,MatrixSize,MinValue,MaxValue,Time(mcs)\n";

    stats << threadsX << "," << threadsY << "," << N << "," << minVal << "," << maxVal << "," << elapsedTime << "\n";
    stats.close();

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
