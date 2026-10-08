#include "../include/AnomalyDetection.h"

// Main api call
int processNewDataPoint(float newValue, float* tde, float* slidingWindow, float* runningMean,
                        float* runningCov, float* runningScatter,float* principalComponent1, float* principalComponent2, int* indexes,
                        int windowSize, int dimensions, float* outX, float* outY)
{
    if (PCA(runningMean, runningCov, runningScatter,tde, slidingWindow, dimensions, windowSize, newValue, indexes) == 1)
    {
        return 1;
    }

    subspaceIteration(runningCov, dimensions, principalComponent1, principalComponent2);

    *outX = dotProduct(tde,principalComponent1,dimensions);
    *outY = dotProduct(tde,principalComponent2,dimensions);

    return 0;
}

void slideWindow(float* slidingWindow, int size, float value)
{
    for (int i = 0; i < size - 1; i++)
    {
        slidingWindow[i] = slidingWindow[i + 1];
    }
    slidingWindow[size - 1] = value;
}

void embeddingIndexes(int* buffer, int windowSize, int dimensions, int tau)
{
    int w = windowSize - 1;
    for (int i = 0; i < dimensions; i++)
    {
        int xt = w - i * tau;
        buffer[i] = xt;
    }
}

void embedding(float* buffer, float* slidingWindow, int size, int* indexes)
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
q2: Point along the second longest axis while staying strictly 90 degree to the fist component.
*/

void subspaceIteration(const float* runningCov, int dim, float* q1, float* q2)
{
    //for later comparison between the original directions and the resulting point
    //(to ensure that the orientation is locked)
    float q1_old[dim];
    float q2_old[dim];

    copyArray(q1,q1_old,dim);
    copyArray(q2,q2_old,dim);

    // Z = C * Q
    //if a random vector is multiplied by C the matrix rotates and stretches toward
    // the direction of maximum variance (largest eigenvector)
    float z1[dim];
    float z2[dim];

    for (int i = 0; i < dim; i++)
    {
        z1[i] = 0.0f;
        z2[i] = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            float C = runningCov[indexAccessHelper(i,j,dim)];
            z1[i] += C * q1[j];
            z2[i] += C * q2[j];
        }
    }
    //after the stretching and rotating the length is not equal to 1 anymore
    // additionaly c pulls everything toward the first principle component

    //1.0 Normalize the first pc by dividing z1 by its length
    float norm1 = sqrtf(dotProduct(z1, z1, dim));
    if (norm1 > 1e-5f)
    {
        for (int i = 0; i < dim; i++)
        {
            q1[i] = z1[i] / norm1;
        }
    }

    //force z2 to be 90 degree onto z1
    float proj = dotProduct(q1, z2, dim);
    for (int i = 0; i < dim; i++) z2[i] -= proj * q1[i];

    //normalization of q2
    float norm2 = sqrtf(dotProduct(z2, z2, dim));
    if (norm2 > 1e-5f)
    {
        for (int i = 0; i < dim; i++) {
            q2[i] = z2[i] / norm2;
        }
    }

    // compare with original directions to lock the orientation
    if (dotProduct(q1, q1_old, dim) < 0.0f) {
        for (int i = 0; i < dim; i++) q1[i] = -q1[i];
    }
    if (dotProduct(q2, q2_old, dim) < 0.0f) {
        for (int i = 0; i < dim; i++) q2[i] = -q2[i];
    }
}

int PCA(float* runningMean, float* runningCov, float* runningScatter,float* tde, float* slidingWindow, int dimensions,
        int windowSize, float newValue, int* indexes)
{
    static int sampleCount = 0; // reusing in C++/Java frage wichtig static

    int tau = indexes[0] - indexes[1];
    bool isWindowFull = (sampleCount >= windowSize);

    float tdeOldRaw[dimensions];
    float tdeOldCentered[dimensions];
    float* oldCenteredAddress = NULL;
    float* oldRawAddress = NULL;

    if (isWindowFull)
    {
        embedding(tdeOldRaw,slidingWindow,dimensions,indexes);
        centerData(runningMean, tdeOldRaw, tdeOldCentered, dimensions);
        oldCenteredAddress = tdeOldCentered;
        oldRawAddress = tdeOldRaw;
    }

    slideWindow(slidingWindow, windowSize, newValue);
    sampleCount++;

    int minSamples = (dimensions - 1) * tau + 1;
    if (sampleCount < minSamples)
    {
        return 1;
    }

    int sampleSize = isWindowFull ? (windowSize - (dimensions - 1) * tau) : (sampleCount - (dimensions - 1) * tau);

    embedding(tde, slidingWindow, dimensions, indexes);

    float tdeCenteredOldMean[dimensions];
    if (!isWindowFull) {
        centerData(runningMean,tde,tdeCenteredOldMean,dimensions);
    }

    updateMean(runningMean, dimensions, sampleSize, tde, oldRawAddress);
    centerData(runningMean, tde, tde, dimensions);

    if (isWindowFull) {
        updateCovariance(runningCov,dimensions,tde, oldCenteredAddress,sampleSize);
    }else {
        updateCovarianceIncremental(runningScatter, dimensions, tdeCenteredOldMean, tde, sampleSize);
        scatterToCovariance(runningScatter, runningCov, dimensions, sampleSize);
    }

    return 0;
}

void updateMean(float* runningMean, int dimensions, int sampleSize, const float* newEmbedded, const float* oldEmbedded)
{
    for (int i = 0; i < dimensions; i++)
    {
        if (oldEmbedded != NULL)
        {
            // Fixed window size
            runningMean[i] += (newEmbedded[i] - oldEmbedded[i]) / (float)sampleSize;
        }
        else
        {
            // Growing window size
            runningMean[i] += (newEmbedded[i] - runningMean[i]) / (float)sampleSize;
        }
    }
}

void centerData(float* mean, float* tdeIn, float* tdeOut, int dimensions)
{
    for (int i = 0; i < dimensions; i++)
    {
        tdeOut[i] = tdeIn[i] - mean[i];
    }
}

void copyArray(float* inputArray, float* outputArray, int dimensions)
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

void updateCovariance(float* runningCov, int dimensions, const float* newCentered,
                      const float* oldCentered, int sampleSize)
{
    if (sampleSize <= 1) return;

    for (int i = 0; i < dimensions; i++)
    {
        for (int j = 0; j < dimensions; j++)
        {
            int covIdx = indexAccessHelper(i, j, dimensions);
            float contribution = (newCentered[i] * newCentered[j]) - (oldCentered[i] * oldCentered[j]);
            runningCov[covIdx] += contribution / (float)sampleSize;
        }
    }
}

void updateCovarianceIncremental(float* runningScatter, int dimensions,
                                  const float* deltaOld, const float* deltaNew, int sampleSize)
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

void scatterToCovariance(const float* scatter, float* covOut, int dimensions, int sampleSize)
{
    float denom = (sampleSize > 1) ? (float)(sampleSize - 1) : 1.0f;
    for (int k = 0; k < dimensions * dimensions; k++)
    {
        covOut[k] = scatter[k] / denom;
    }
}

float dotProduct(const float* v1, const float* v2, int dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        sum += v1[i] * v2[i];
    }
    return sum;
}