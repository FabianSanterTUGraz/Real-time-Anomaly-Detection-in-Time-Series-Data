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
    int processNewDataPoint(double newValue, double* tde, double* slidingWindow,
                            double* runningMean, double* runningCov,
                            double* runningScatter, double* eigenvalues,
                            double* eigenvectors, int* indexes, int windowSize,
                            int dimensions, double* outX, double* outY);

    void slideWindow(double* slidingWindow, int size, double value);

    void embedding(double* buffer, double* slidingWindow, int size, int* indexes);

    void embeddingIndexes(int* buffer, int windowSize, int dimensions);

    int PCA(double* runningMean, double* runningCov, double* runningScatter, double* tde,
            double* slidingWindow, int dimensions, int windowSize, double newValue,
            int* indexes);

    void updateMean(double* runningMean, int dimensions, int n,  double* newEmbedded,
                     double* oldEmbedded);

    void centerData(double* runningMean, double* tdeIn, double* tdeOut, int dimensions);

    void copyArray(double* inputArray, double* outputArray, int dimensions);

    int indexAccessHelper(int row, int column, int dimensions);

    double dotProduct( double* v1,  double* v2, int dim);

    void subspaceIteration( double* runningCov, int dimensions, double* tde1,
                           double* tde2);

    // tmp:
    void updateCovarianceIncremental(double* runningScatter, int dimensions,
                                      double* deltaOld,  double* deltaNew);

    void slideScatter(double* runningScatter, int dimensions,
                       double* newCenteredOldMean,  double* newCenteredNewMean,
                       double* oldCenteredOldMean,  double* oldCenteredNewMean);

    void normalize(double* z1, double* z2, int dim, double* q1, double* q2);
    void initLagRing( double* runningScatter,  double* runningMean,
                     int dimensions, int vectorsInWindow, int ringPos, double* lagRing);
    void slideLagRing(double* lagRing, int dimensions, int ringPos,  double* newRaw,
                       double* oldRaw);
    void lagMatVec( double* lagRing,  double* runningMean, int dimensions,
                   int vectorsInWindow, int ringPos,  double* q1,  double* q2,
                   double* z1, double* z2);
#ifdef __cplusplus
}
#endif

#endif  // ANOMALYDETECTION_H