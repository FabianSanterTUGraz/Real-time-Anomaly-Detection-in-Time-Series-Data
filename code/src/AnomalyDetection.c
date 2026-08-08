#include "../include/AnomalyDetection.h"

#define MAX_ITERATIONS 100
#define EPSILON 1e-6f

// Main api calls
int processNewDataPoint(float newValue, float* tde, float* slidingWindow, float* runningMean,
                        float* runningCov, float* eigenvalues, float* eigenvectors, int* indexes,
                        int windowSize, int dimensions, int* sampleCount, float* outX, float* outY)
{
    int enoughtInformation =
        PCA(runningMean, runningCov, tde, slidingWindow, dimensions, windowSize, newValue, indexes);
    if (enoughtInformation == 0)
    {
        return 1;
    }

    int tau = indexes[0] - indexes[1];

    jacobiEigenvalue(runningCov, dimensions, eigenvalues, eigenvectors);
    int principalComponent1 = 0;
    int principalComponent2 = 0;

    findTopTwoComponents(eigenvalues, dimensions, &principalComponent1, &principalComponent2);

    projectData(tde, eigenvectors, dimensions, principalComponent1, outX);
    projectData(tde, eigenvectors, dimensions, principalComponent2, outY);

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

void jacobiEigenvalue(float* runningCov, int dim, float* eigenvalues, float* eigenvectors)
{
    // 1. Initialize Eigenvectors as Identity Matrix
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            // set diagonale to 1 and off diagonal to 0.
            eigenvectors[indexAccessHelper(i, j, dim)] = (i == j) ? 1.0f : 0.0f;
        }
    }

    float A[dim * dim];
    copyArray(runningCov, A, dim * dim);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        // Find the largest off-diagonal element A[p][q]
        int p = 0, q = 1;
        float max_val = fabs(A[indexAccessHelper(0, 1, dim)]); // or std::abs(),Math.abs()

        for (int i = 0; i < dim; i++)
        {
            for (int j = i + 1; j < dim; j++)
            {
                float val = fabs(A[indexAccessHelper(i, j, dim)]);
                if (val > max_val)
                {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Check Cif value converges to epsilon
        if (max_val < EPSILON)
        {
            break;
        }

        float app = A[indexAccessHelper(p, p, dim)];
        float aqq = A[indexAccessHelper(q, q, dim)];
        float apq = A[indexAccessHelper(p, q, dim)];

        float theta = (aqq - app) / (2.0f * apq);
        float abs_theta = fabs(theta);

        // c++ : std::sqrt(), java: Math.sqrt()
        float t = 1.0f / (abs_theta + sqrt(theta * theta + 1.0f));
        if (theta < 0.0f)
        {
            t = -t;
        }

        float c = 1.0f / sqrt(t * t + 1.0f);
        float s = t * c;

        float app_new = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        float aqq_new = s * s * app + 2.0f * s * c * apq + c * c * aqq;

        A[indexAccessHelper(p, p, dim)] = app_new;
        A[indexAccessHelper(q, q, dim)] = aqq_new;
        A[indexAccessHelper(p, q, dim)] = 0.0f;
        A[indexAccessHelper(q, p, dim)] = 0.0f;

        for (int i = 0; i < dim; i++)
        {
            if (i != p && i != q)
            {
                float aip = A[indexAccessHelper(i, p, dim)];
                float aiq = A[indexAccessHelper(i, q, dim)];
                A[indexAccessHelper(i, p, dim)] = c * aip - s * aiq;
                A[p * dim + i] = c * aip - s * aiq;
                A[indexAccessHelper(i, q, dim)] = s * aip + c * aiq;
                A[q * dim + i] = s * aip + c * aiq;
            }
        }

        for (int i = 0; i < dim; i++)
        {
            float vip = eigenvectors[indexAccessHelper(i, p, dim)];
            float viq = eigenvectors[indexAccessHelper(i, q, dim)];
            eigenvectors[indexAccessHelper(i, p, dim)] = c * vip - s * viq;
            eigenvectors[indexAccessHelper(i, q, dim)] = s * vip + c * viq;
        }
    }

    for (int i = 0; i < dim; i++)
    {
        eigenvalues[i] = A[indexAccessHelper(i, i, dim)];
    }
}

void findTopTwoComponents(const float* eigenvalues, int dim, int* idx_pc1, int* idx_pc2)
{
    int first = 0, second = -1;
    float max1 = eigenvalues[0];
    float max2 = -1e9f;

    for (int i = 1; i < dim; i++)
    {
        if (eigenvalues[i] > max1)
        {
            max2 = max1;
            second = first;
            max1 = eigenvalues[i];
            first = i;
        }
        else if (eigenvalues[i] > max2)
        {
            max2 = eigenvalues[i];
            second = i;
        }
    }
    *idx_pc1 = first;
    *idx_pc2 = second;
}

void projectData(float* tde, float* eigenvectors, int dimensions, int targetComponentIdx,
                 float* outputProjection)
{
    float sum = 0.0f;
    for (int i = 0; i < dimensions; i++)
    {
        int matrixIndex = indexAccessHelper(i, targetComponentIdx,
                                            dimensions); // i * dimensions + targetComponentIdx;
        sum += tde[i] * eigenvectors[matrixIndex];
    }
    *outputProjection = sum;
}

int validEmbeddingCount(int sampleCount, int dimensions, int windowSize, int tau)
{
    int capacity = windowSize - (dimensions - 1) * tau;
    int rawValidCount = sampleCount - (dimensions - 1) * tau;
    return rawValidCount > capacity ? capacity : rawValidCount;
}

int PCA(float* runningMean, float* runningCov, float* tde, float* slidingWindow, int dimensions,
        int windowSize, float newValue, int* indexes)
{
    static int sampleCount = 0;
    int tau = indexes[0] - indexes[1]; // Deduce tau spacing dynamically
    float tdeOldRaw[dimensions];
    for (int i = 0; i < dimensions; i++)
    {
        tdeOldRaw[i] = slidingWindow[(dimensions - 1 - i) * tau];
    }

    slideWindow(slidingWindow, windowSize, newValue);
    sampleCount++;

    if (sampleCount < dimensions)
    {
        return 0; // not enough raw history yet for a single valid embedded vector
    }
    int n = sampleCount - (dimensions - 1) * tau;

    int rawValidCount = sampleCount - (dimensions - 1) * tau;

    embedding(tde, slidingWindow, dimensions, indexes); // raw, uncentered new embedded vector

    // tdeOldRaw belongs to the step BEFORE this one, so it must be centered with the mean
    // as it was before updateMean runs below, not with the just-updated mean.
    float tdeOldCentered[dimensions];

    centerData(runningMean, tdeOldRaw, tdeOldCentered, dimensions);

    updateMean(runningMean, dimensions, n, tde);

    centerData(runningMean, tde, tde, dimensions);

    updateCovariance(runningCov, dimensions, tde, NULL);

    return 1;
}

void updateMean(float* runningMean, int dimensions, int n, const float* newEmbedded)
{
    // Window full: evict the oldest embedded vector, add the newest; n stays at capacity.
    for (int i = 0; i < dimensions; i++)
    {
        runningMean[i] += (newEmbedded[i] - runningMean[i]) / (float)n;
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
                      const float* oldCentered)
{
    for (int i = 0; i < dimensions; i++)
    {
        for (int j = 0; j < dimensions; j++)
        {
            int covIdx = indexAccessHelper(i, j, dimensions);
            float contribution = newCentered[i] * newCentered[j];
            if (oldCentered != NULL)
            {
                contribution -= oldCentered[i] * oldCentered[j];
            }
            runningCov[covIdx] += contribution;
        }
    }
}