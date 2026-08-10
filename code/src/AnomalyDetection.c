#include "../include/AnomalyDetection.h"

// Main api call
int processNewDataPoint(float newValue, float* tde, float* slidingWindow, float* runningMean,
                        float* runningCov, float* principalComponent1, float* principalComponent2, int* indexes,
                        int windowSize, int dimensions, float* outX, float* outY)
{
    if (PCA(runningMean, runningCov, tde, slidingWindow, dimensions, windowSize, newValue, indexes) == 1)
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

void embedding(float* buffer, float* slidingWindow, int size, int* indexes)
{
    for (int i = 0; i < size; i++)
    {
        int xt = indexes[i];
        buffer[i] = slidingWindow[xt];
    }
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

void subspaceIteration(const float* runningCov, int dim, float* q1, float* q2)
{
    float q1_old[dim];
    float q2_old[dim];
    
    copyArray(q1,q1_old,dim);
    copyArray(q2,q2_old,dim);

    // Step 1: Matrix multiplication Z = C * Q
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

    float norm1 = sqrtf(dotProduct(z1, z1, dim));
    if (norm1 > 1e-6f) {
        for (int i = 0; i < dim; i++) q1[i] = z1[i] / norm1;
    }

    float proj = dotProduct(q1, z2, dim);
    for (int i = 0; i < dim; i++) z2[i] -= proj * q1[i];

    float norm2 = sqrtf(dotProduct(z2, z2, dim));
    if (norm2 > 1e-6f) {
        for (int i = 0; i < dim; i++) q2[i] = z2[i] / norm2;
    }

    // 4. Sign-Locking (Prevents 180-degree flipping)
    if (dotProduct(q1, q1_old, dim) < 0.0f) {
        for (int i = 0; i < dim; i++) q1[i] = -q1[i];
    }
    if (dotProduct(q2, q2_old, dim) < 0.0f) {
        for (int i = 0; i < dim; i++) q2[i] = -q2[i];
    }
}

int PCA(float* runningMean, float* runningCov, float* tde, float* slidingWindow, int dimensions,
        int windowSize, float newValue, int* indexes)
{
    static int sampleCount = 0; // reusing in C++/Java frage 

    int tau = indexes[0] - indexes[1];
    bool isWindowFull = (sampleCount >= windowSize);

    float tdeOldRaw[dimensions];
    float tdeOldCentered[dimensions];
    float* oldCenteredAddress = NULL;
    float* oldRawAddress = NULL;

    if (isWindowFull)
    {
        copyArray(tde,tdeOldRaw,dimensions);
        centerData(runningMean, tdeOldRaw, tdeOldCentered, dimensions);
        oldCenteredAddress = tdeOldCentered;
        oldRawAddress = tdeOldRaw;
    }

    // 2. Slide window with new value
    slideWindow(slidingWindow, windowSize, newValue);
    sampleCount++;

    // 3. Warm-up check
    int minSamples = (dimensions - 1) * tau + 1;
    if (sampleCount < minSamples)
    {
        return 1;
    }

    int sampleSize = isWindowFull ? (windowSize - (dimensions - 1) * tau) : (sampleCount - (dimensions - 1) * tau);

    embedding(tde, slidingWindow, dimensions, indexes);
    updateMean(runningMean, dimensions, sampleSize, tde, oldRawAddress);
    centerData(runningMean, tde, tde, dimensions); 
    updateCovariance(runningCov, dimensions, tde, oldCenteredAddress, sampleSize);

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
            
            if (oldCentered != NULL)
            {
                // Sliding window (fixed n): Add new, subtract old
                float contribution = (newCentered[i] * newCentered[j]) - (oldCentered[i] * oldCentered[j]);
                runningCov[covIdx] += contribution / (float)sampleSize;
            }
            else
            {
                // Growing window (increasing n): Running average update
                float contribution = newCentered[i] * newCentered[j];
                runningCov[covIdx] += (contribution - runningCov[covIdx]) / (float)sampleSize;
            }
        }
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