#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <string>
#include <chrono>
#include <omp.h>

std::vector<std::vector<int>> readMatrix(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    std::vector<std::vector<int>> matrix;
    int value;
    std::string line;

    while (std::getline(file, line)) {
        std::vector<int> row;
        std::stringstream ss(line);
        while (ss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }

    return matrix;
}

void writeMatrix(const std::string& filePath, const std::vector<std::vector<int>>& matrix) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << " ";
            }
        }
        file << "\n";
    }
}

std::vector<std::vector<int>> multiplyMatrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    size_t colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrixes with such dimensions cannot be multiplied");
    }

    std::vector<std::vector<int>> result(rowsA, std::vector<int>(colsB, 0));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            int sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < colsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

int main(int argc, char* argv[]) {

    std::cout << omp_get_max_threads();

    if (argc != 5) {
        std::cerr << "Arguments: " << argv[0] << "<path to first> <path to second> <path to result>" << std::endl;
        return 1;
    }

    omp_set_num_threads(std::stoi(argv[4]));

    std::vector<std::vector<int>> matrixA = readMatrix(argv[1]);
    std::vector<std::vector<int>> matrixB = readMatrix(argv[2]);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> result = multiplyMatrices(matrixA, matrixB);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    writeMatrix(argv[3], result);

    std::cout << "Correctly saved to: " << argv[3] << std::endl;

    return static_cast<int>(duration.count());
}