#include "../include/AnomalyDetection.h"

// Main api call
static int SAMPLES_TO_NEXT_ITERATION = 200;
static int counter = 0;
static int sampleCount = 0;

int processNewDataPoint(double newValue, double* tde, double* slidingWindow,
                        double* runningMean, double* runningCov, double* runningScatter,
                        double* principalComponent1, double* principalComponent2,
                        int* indexes, int windowSize, int dimensions, double* outX,
                        double* outY)
{
    if (PCA(runningMean, runningCov, runningScatter, tde, slidingWindow, dimensions,
            windowSize, newValue, indexes) == 1)
    {
        return 1;
    }

    //No need for using the expensive subspace iteration each time only use up to a specific threshold
    if (counter > SAMPLES_TO_NEXT_ITERATION)
    {
        counter = 0;
        if (sampleCount > windowSize)
        {
            double z1[dimensions];
            double z2[dimensions];
            int vectorsInWindow = windowSize - (dimensions - 1);
            lagMatVec(runningCov, runningMean, dimensions, vectorsInWindow,
                      sampleCount % dimensions, principalComponent1, principalComponent2,
                      z1, z2);
            normalize(z1, z2, dimensions, principalComponent1,
                                  principalComponent2);
        }
        else
        {
            subspaceIteration(runningScatter, dimensions, principalComponent1,
                              principalComponent2);
        }
    }
    counter++;

    *outX = dotProduct(tde, principalComponent1, dimensions);
    *outY = dotProduct(tde, principalComponent2, dimensions);

    return 0;
}

void slideWindow(double* slidingWindow, int size, double value)
{
    for (int i = 0; i < size - 1; i++)
    {
        slidingWindow[i] = slidingWindow[i + 1];
    }
    slidingWindow[size - 1] = value;
}

void embeddingIndexes(int* buffer, int windowSize, int dimensions)
{
    int tau = 1;
    int w = windowSize - 1;
    for (int i = 0; i < dimensions; i++)
    {
        int xt = w - i * tau;
        buffer[i] = xt;
    }
}

void embedding(double* buffer, double* slidingWindow, int size, int* indexes)
{
    for (int i = 0; i < size; i++)
    {
        int xt = indexes[i];
        buffer[i] = slidingWindow[xt];
    }
}

/*
Issue which is solved by this function:
Input in a higher dimensional point cloud of streaming data and the
algorithm should find the two main directions (axes) along which the
data varies the most.
q1: Point along the absolute axis of the data cloud. 1st principal component
q2: Point along the second longest axis while staying strictly 90 degree to the fist
component.
*/

void subspaceIteration( double* runningCov, int dim, double* q1, double* q2)
{
    // Z = C * Q
    // if a random vector is multiplied by C the matrix rotates and stretches toward
    // the direction of maximum variance (largest eigenvector)
    double z1[dim];
    double z2[dim];

    for (int i = 0; i < dim; i++)
    {
        z1[i] = 0.0f;
        z2[i] = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            double C = runningCov[indexAccessHelper(i, j, dim)];
            z1[i] += C * q1[j];
            z2[i] += C * q2[j];
        }
    }
    normalize(z1, z2, dim, q1, q2);
}

void normalize(double* z1, double* z2, int dim, double* q1, double* q2)
{
    // for later comparison between the original directions and the resulting point
    //(to ensure that the orientation is locked)
    double q1_old[dim];
    double q2_old[dim];

    copyArray(q1, q1_old, dim);
    copyArray(q2, q2_old, dim);


    // after the stretching and rotating the length is not equal to 1 anymore
    //  additionaly c pulls everything toward the first principle component

    // 1.0 Normalize the first pc by dividing z1 by its length
    double norm1 = sqrt(dotProduct(z1, z1, dim));
    if (norm1 > 1e-5f)
    {
        for (int i = 0; i < dim; i++)
        {
            q1[i] = z1[i] / norm1;
        }
    }

    // force z2 to be 90 degree onto z1
    double proj = dotProduct(q1, z2, dim);
    for (int i = 0; i < dim; i++)
    {
        z2[i] -= proj * q1[i];
    }

    // normalization of q2
    double norm2 = sqrt(dotProduct(z2, z2, dim));
    if (norm2 > 1e-5f)
    {
        for (int i = 0; i < dim; i++)
        {
            q2[i] = z2[i] / norm2;
        }
    }

    // compare with original directions to lock the orientation
    if (dotProduct(q1, q1_old, dim) < 0.0f)
    {
        for (int i = 0; i < dim; i++)
        {
            q1[i] = -q1[i];
        }
    }
    if (dotProduct(q2, q2_old, dim) < 0.0f)
    {
        for (int i = 0; i < dim; i++)
        {
            q2[i] = -q2[i];
        }
    }
}

int PCA(double* runningMean, double* runningCov, double* runningScatter, double* tde,
        double* slidingWindow, int dimensions, int windowSize, double newValue,
        int* indexes)
{
    int tau = 1;
    bool isWindowFull = (sampleCount >= windowSize);
    bool useLagRing = isWindowFull;

    double tdeOldRaw[dimensions];
    double tdeOldCentered[dimensions];
    double* oldCenteredAddress = NULL;
    double* oldRawAddress = NULL;

    if (isWindowFull)
    {
        int shift = windowSize - 1 - (dimensions - 1) * tau;
        for (int i = 0; i < dimensions; i++)
        {
            tdeOldRaw[i] = slidingWindow[indexes[i] - shift];
        }
        centerData(runningMean, tdeOldRaw, tdeOldCentered, dimensions);
        oldCenteredAddress = tdeOldCentered;
        oldRawAddress = tdeOldRaw;
    }

    // NEU: switch - the window became full with the previous sample.
    // Fill the lag ring ONCE from scatter + mean; from now on runningCov holds the lag
    // ring.
    if (useLagRing && sampleCount == windowSize)
    {
        int vectorsInWindow = windowSize - (dimensions - 1);
        initLagRing(runningScatter, runningMean, dimensions, vectorsInWindow,
                    sampleCount % dimensions, runningCov);
    }

    slideWindow(slidingWindow, windowSize, newValue);
    sampleCount++;

    int minSamples = (dimensions - 1) * tau + 1;
    if (sampleCount < minSamples)
    {
        return 1;
    }

    int sampleSize = isWindowFull ? (windowSize - (dimensions - 1) * tau)
                                  : (sampleCount - (dimensions - 1) * tau);

    embedding(tde, slidingWindow, dimensions, indexes);

    // NEU: sliding phase with tau = 1 -> O(d), the scatter matrix is not touched anymore
    if (useLagRing)
    {
        slideLagRing(runningCov, dimensions, sampleCount % dimensions, tde,
                     tdeOldRaw);  // tde is still raw here!
        updateMean(runningMean, dimensions, sampleSize, tde, oldRawAddress);
        centerData(runningMean, tde, tde, dimensions);
        return 0;
    }

    // filling phase (and sliding phase for tau != 1): unchanged
    double tdeCenteredOldMean[dimensions];
    centerData(runningMean, tde, tdeCenteredOldMean, dimensions);

    updateMean(runningMean, dimensions, sampleSize, tde, oldRawAddress);
    centerData(runningMean, tde, tde, dimensions);

    if (isWindowFull)
    {
        double oldCenteredNewMean[dimensions];
        centerData(runningMean, tdeOldRaw, oldCenteredNewMean, dimensions);
        slideScatter(runningScatter, dimensions, tdeCenteredOldMean, tde,
                     oldCenteredAddress, oldCenteredNewMean);
    }
    else
    {
        updateCovarianceIncremental(runningScatter, dimensions, tdeCenteredOldMean, tde);
    }

    return 0;
}

void updateMean(double* runningMean, int dimensions, int sampleSize,
                 double* newEmbedded,  double* oldEmbedded)
{
    for (int i = 0; i < dimensions; i++)
    {
        if (oldEmbedded != NULL)
        {
            // Fixed window size
            runningMean[i] += (newEmbedded[i] - oldEmbedded[i]) / (double)sampleSize;
        }
        else
        {
            // Growing window size
            runningMean[i] += (newEmbedded[i] - runningMean[i]) / (double)sampleSize;
        }
    }
}

void centerData(double* mean, double* tdeIn, double* tdeOut, int dimensions)
{
    for (int i = 0; i < dimensions; i++)
    {
        tdeOut[i] = tdeIn[i] - mean[i];
    }
}

void copyArray(double* inputArray, double* outputArray, int dimensions)
{
    for (int i = 0; i < dimensions; i++)
    {
        outputArray[i] = inputArray[i];
    }
}

int indexAccessHelper(int row, int column, int dimensions)
{
    return (row * dimensions) + column;
}

void updateCovarianceIncremental(double* runningScatter, int dimensions,
                                  double* deltaOld,  double* deltaNew)
{
    for (int i = 0; i < dimensions; i++)
    {
        for (int j = 0; j < dimensions; j++)
        {
            int idx = indexAccessHelper(i, j, dimensions);
            runningScatter[idx] += deltaOld[i] * deltaNew[j];
        }
    }
}

double dotProduct( double* v1,  double* v2, int dim)
{
    double sum = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        sum += v1[i] * v2[i];
    }
    return sum;
}

// add the new vector and remove the old one in ONE pass over the scatter matrix
void slideScatter(double* runningScatter, int dimensions,
                   double* newCenteredOldMean,  double* newCenteredNewMean,
                   double* oldCenteredOldMean,  double* oldCenteredNewMean)
{
    for (int i = 0; i < dimensions; i++)
    {
        double a = newCenteredOldMean[i];
        double c = oldCenteredOldMean[i];
        for (int j = 0; j < dimensions; j++)
        {
            runningScatter[indexAccessHelper(i, j, dimensions)] +=
                a * newCenteredNewMean[j] - c * oldCenteredNewMean[j];
        }
    }
}

/*
Lag ring (only tau = 1). Once the window is full, the array runningCov holds it:
row p = lag sums r_0..r_{d-1} of the window that ended at the sample with sampleCount % d
== p, r_k(t) = sum over window of x[s] * x[s-k] Every entry of the (unnormalized) matrix
follows from it:   R_ij(t) = r_{j-i}(t-i)   (j >= i)
*/

// once, when the window is full:  r_k(t0 - i) = R[i][i+k] = S[i][i+k] + N * mean[i] *
// mean[i+k]
void initLagRing( double* runningScatter,  double* runningMean, int dimensions,
                 int vectorsInWindow, int ringPos, double* lagRing)
{
    for (int i = 0; i < dimensions; i++)
    {
        int row = ((ringPos - i) % dimensions + dimensions) % dimensions;
        for (int k = 0; k < dimensions; k++)
        {
            int j = i + k;
            if (j < dimensions)
            {
                lagRing[indexAccessHelper(row, k, dimensions)] =
                    runningScatter[indexAccessHelper(i, j, dimensions)] +
                    vectorsInWindow * runningMean[i] * runningMean[j];
            }
            else
            {
                lagRing[indexAccessHelper(row, k, dimensions)] = 0.0;  // never read
            }
        }
    }
}

// every sample:  r_k(t) = r_k(t-1) + x[t] * x[t-k] - x[old] * x[old-k]
// newRaw[k] = x[t-k] (raw tde, NOT centered), oldRaw[k] = x[old-k] (raw tde leaving the
// window)
void slideLagRing(double* lagRing, int dimensions, int ringPos,  double* newRaw,
                   double* oldRaw)
{
    int prev = (ringPos - 1 + dimensions) % dimensions;
    double a = newRaw[0];
    double c = oldRaw[0];
    for (int k = 0; k < dimensions; k++)
    {
        lagRing[indexAccessHelper(ringPos, k, dimensions)] =
            lagRing[indexAccessHelper(prev, k, dimensions)] + a * newRaw[k] -
            c * oldRaw[k];
    }
}

// z = S * q  with  S = R - N * mean * mean^T  (same scale as the scatter matrix), R read
// from the ring
void lagMatVec( double* lagRing,  double* runningMean, int dimensions,
               int vectorsInWindow, int ringPos,  double* q1,  double* q2,
               double* z1, double* z2)
{
    for (int i = 0; i < dimensions; i++) { z1[i] = 0.0; }
    for (int i = 0; i < dimensions; i++) { z2[i] = 0.0; }


    for (int i = 0; i < dimensions; i++)
    {
        // row i of R from the diagonal on:  R_ij = r_{j-i}(t-i)
        int row = ((ringPos - i) % dimensions + dimensions) % dimensions;
        double Rii = lagRing[indexAccessHelper(row, 0, dimensions)];
        z1[i] += Rii * q1[i];
        z2[i] += Rii * q2[i];
        for (int j = i + 1; j < dimensions; j++)
        {
            double R = lagRing[indexAccessHelper(row, j - i, dimensions)];
            z1[i] += R * q1[j];  // upper triangle
            z2[i] += R * q2[j];
            z1[j] += R * q1[i];  // mirrored lower triangle
            z2[j] += R * q2[i];
        }
    }
    // subtract the mean part:  N * mean * (mean^T q)
    double mq1 = dotProduct(runningMean, q1, dimensions);
    double mq2 = dotProduct(runningMean, q2, dimensions);
    for (int i = 0; i < dimensions; i++)
    {
        z1[i] -= vectorsInWindow * runningMean[i] * mq1;
        z2[i] -= vectorsInWindow * runningMean[i] * mq2;
    }
}