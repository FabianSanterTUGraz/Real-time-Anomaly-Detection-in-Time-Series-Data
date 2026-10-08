#ifndef ANOMALYDETECTION_H
#define ANOMALYDETECTION_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif
    int processNewDataPoint(double newValue, double* tde, double* slidingWindow, double* runningMean,
                            double* runningCov, double* runningScatter,double* eigenvalues, double* eigenvectors,
                            int* indexes, int windowSize, int dimensions,
                            double* outX, double* outY);

    void slideWindow(double* slidingWindow, int size, double value);

    void embedding(double* buffer, double* slidingWindow, int size, int* indexes);

    void embeddingIndexes(int* buffer, int windowSize, int dimensions, int tau);

    int PCA(double* runningMean, double* runningCov, double* runningScatter,double* tde, double* slidingWindow, int dimensions,
            int windowSize, double newValue, int* indexes);

    void updateMean(double* runningMean, int dimensions, int n, const double* newEmbedded, const double* oldEmbedded);

    void centerData(double* runningMean, double* tdeIn, double* tdeOut, int dimensions);

    void copyArray(double* inputArray, double* outputArray, int dimensions);

    int indexAccessHelper(int row, int column, int dimensions);

    void updateCovariance(double* runningCov, int dimensions, const double* newCentered,
                      const double* oldCentered, int n);

    double dotProduct(const double* v1, const double* v2, int dim);

    void subspaceIteration(const double* runningCov,int dimensions, double* tde1, double* tde2);

    //tmp:
    void updateCovarianceIncremental(double* runningScatter, int dimensions,
                                  const double* deltaOld, const double* deltaNew, int sampleSize);

    void scatterToCovariance(const double* scatter, double* covOut, int dimensions, int sampleSize);
#ifdef __cplusplus
}
#endif

#endif // ANOMALYDETECTION_H