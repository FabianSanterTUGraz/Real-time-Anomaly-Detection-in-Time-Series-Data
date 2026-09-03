#include <iostream>
#include "include/AnomalyDetection.h"

int main() {
    const int dimensions = 2;
    const int inputSize = 10;

    float data[10][2] = {
        {2.5f, 2.4f}, {0.5f, 0.7f}, {2.2f, 2.9f}, {1.9f, 2.2f},
        {3.1f, 3.0f}, {2.3f, 2.7f}, {2.0f, 1.6f}, {1.0f, 1.1f},
        {1.5f, 1.6f}, {1.1f, 0.9f}
    };

    float runningMean[dimensions] = {0.0f};
    float runningScatter[dimensions * dimensions] = {0.0f};
    float runningCov[dimensions * dimensions] = {0.0f};

    for (int n = 1; n <= inputSize; ++n) {
        float point[dimensions] = {data[n - 1][0], data[n - 1][1]};
        float deltaOld[dimensions];
        float deltaNew[dimensions];

        centerData(runningMean, point, deltaOld, dimensions);

        updateMean(runningMean, dimensions, n, point, nullptr);

        centerData(runningMean, point, deltaNew, dimensions);

        updateCovarianceIncremental(runningScatter, dimensions, deltaOld, deltaNew, n);
    }

    scatterToCovariance(runningScatter, runningCov, dimensions, inputSize);

    std::cout << "Running mean: " << runningMean[0] << ", " << runningMean[1] << std::endl;
    std::cout << "Running cov: " << runningCov[0] << ", " << runningCov[1]
               << ", " << runningCov[2] << ", " << runningCov[3] << std::endl;

    // Subspace iteration (power method) to extract eigenvectors
    float pc1[dimensions] = {1.0f, 0.0f};
    float pc2[dimensions] = {0.0f, 1.0f};

    for (int iter = 0; iter < 200; ++iter) {
        subspaceIteration(runningCov, dimensions, pc1, pc2);
    }
    std::cout << "PC1: " << pc1[0] << ", " << pc1[1] << std::endl;
    std::cout << "PC2: " << pc2[0] << ", " << pc2[1] << std::endl;

    // Project all points
    std::cout << "\nProjected scores:" << std::endl;
    for (int n = 0; n < inputSize; ++n) {
        float centered[dimensions];
        centerData(runningMean, data[n], centered, dimensions);
        float x = dotProduct(centered, pc1, dimensions);
        float y = dotProduct(centered, pc2, dimensions);
        std::cout << x << ", " << y << std::endl;
    }

    return 0;
}