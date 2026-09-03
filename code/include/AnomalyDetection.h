#ifndef ANOMALYDETECTION_H
#define ANOMALYDETECTION_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "../include/utils.h"

#ifdef __cplusplus
extern "C"
{
#endif
    int processNewDataPoint(float newValue, float* tde, float* slidingWindow, float* runningMean,
                            float* runningCov, float* runningScatter,float* eigenvalues, float* eigenvectors,
                            int* indexes, int windowSize, int dimensions,
                            float* outX, float* outY);

    void slideWindow(float* slidingWindow, int size, float value);

    void embedding(float* buffer, float* slidingWindow, int size, int* indexes);

    void embeddingIndexes(int* buffer, int windowSize, int dimensions, int tau);

    int PCA(float* runningMean, float* runningCov, float* runningScatter,float* tde, float* slidingWindow, int dimensions,
            int windowSize, float newValue, int* indexes);

    void updateMean(float* runningMean, int dimensions, int n, const float* newEmbedded, const float* oldEmbedded);

    void centerData(float* runningMean, float* tdeIn, float* tdeOut, int dimensions);

    void copyArray(float* inputArray, float* outputArray, int dimensions);

    int indexAccessHelper(int row, int column, int dimensions);

    void updateCovariance(float* runningCov, int dimensions, const float* newCentered,
                      const float* oldCentered, int n);

    float dotProduct(const float* v1, const float* v2, int dim);

    void subspaceIteration(const float* runningCov,int dimensions, float* tde1, float* tde2);

    //tmp:
    void updateCovarianceIncremental(float* runningScatter, int dimensions,
                                  const float* deltaOld, const float* deltaNew, int sampleSize);

    void scatterToCovariance(const float* scatter, float* covOut, int dimensions, int sampleSize);
#ifdef __cplusplus
}
#endif

#endif // ANOMALYDETECTION_H