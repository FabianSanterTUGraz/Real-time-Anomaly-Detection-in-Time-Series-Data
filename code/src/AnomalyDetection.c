#include "../include/AnomalyDetection.h"

#define MAX_ITERATIONS 100
#define EPSILON 1e-6f

// Main api calls
int processNewDataPoint(float newValue, float* tde, float* slidingWindow, float* runningMean,
                        float* runningCov, float* principalComponent1, float* principalComponent2, int* indexes,
                        int windowSize, int dimensions, float* outX, float* outY)
{
    if (PCA(runningMean, runningCov, tde, slidingWindow, dimensions, windowSize, newValue, indexes) == 1)
    {
        return 1;
    }
    float covNormalized[dimensions * dimensions];
    float maxVal = 0.0f;
    for (int i = 0; i < dimensions * dimensions; i++) {
        float absVal = fabsf(runningCov[i]);
        if (absVal > maxVal) maxVal = absVal;
    }

    float scale = (maxVal > 1.0f) ? maxVal : 1.0f;
    for (int i = 0; i < dimensions * dimensions; i++) {
        covNormalized[i] = runningCov[i] / scale;
    }

    //jacobiEigenvalue(runningCov, dimensions, eigenvalues, eigenvectors);
    //int principalComponent1 = 0;
    //int principalComponent2 = 0;

    subspaceIteration(covNormalized, dimensions, principalComponent1, principalComponent2);

    //projectData(tde, eigenvectors, dimensions, principalComponent1, outX);
    //projectData(tde, eigenvectors, dimensions, principalComponent2, outY);
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



void matMul(const float* A, const float* B, float* result, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < dim; k++)
            {
                sum += A[indexAccessHelper(i, k, dim)] * B[indexAccessHelper(k, j, dim)];
            }
            result[indexAccessHelper(i, j, dim)] = sum;
        }
    }
}

void matMulTransposeA(const float* A, const float* B, float* result, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < dim; k++)
            {
                sum += A[indexAccessHelper(k, i, dim)] * B[indexAccessHelper(k, j, dim)];
            }
            result[indexAccessHelper(i, j, dim)] = sum;
        }
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


void subspaceIteration(const float* runningCov, int dim, float* q1, float* q2)
{
    // Store previous vectors for sign continuity
    float q1_old[dim], q2_old[dim];
    for (int i = 0; i < dim; i++) {
        q1_old[i] = q1[i];
        q2_old[i] = q2[i];
    }

    const int ITERATIONS = 3; 
    float z1[dim], z2[dim];

    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        // Step 1: Matrix multiplication Z = C * Q
        for (int i = 0; i < dim; i++)
        {
            z1[i] = 0.0f;
            z2[i] = 0.0f;
            for (int j = 0; j < dim; j++)
            {
                int idx = indexAccessHelper(i, j, dim);
                z1[i] += runningCov[idx] * q1[j];
                z2[i] += runningCov[idx] * q2[j];
            }
        }

        // Step 2: Gram-Schmidt Orthogonalization
        float norm1 = sqrt(dotProduct(z1, z1, dim));
        if (norm1 > 1e-6f) {
            for (int i = 0; i < dim; i++) q1[i] = z1[i] / norm1;
        } else {
            for (int i = 0; i < dim; i++) q1[i] = 0.0f;
            q1[0] = 1.0f;
        }

        float dot12 = dotProduct(q1, z2, dim);
        for (int i = 0; i < dim; i++) z2[i] -= dot12 * q1[i];

        float norm2 = sqrt(dotProduct(z2, z2, dim));
        if (norm2 > 1e-6f) {
            for (int i = 0; i < dim; i++) q2[i] = z2[i] / norm2;
        } else {
            for (int i = 0; i < dim; i++) q2[i] = 0.0f;
            if (dim > 1) q2[1] = 1.0f;
        }
    }

    // Step 3: 2x2 Subspace Diagonalization (Locks rotation angle in 2D plane)
    // Compute A = Q^T * C * Q
    float a11 = 0.0f, a12 = 0.0f, a22 = 0.0f;
    for (int i = 0; i < dim; i++) {
        float C_q1_i = 0.0f, C_q2_i = 0.0f;
        for (int j = 0; j < dim; j++) {
            int idx = indexAccessHelper(i, j, dim);
            C_q1_i += runningCov[idx] * q1[j];
            C_q2_i += runningCov[idx] * q2[j];
        }
        a11 += q1[i] * C_q1_i;
        a12 += q1[i] * C_q2_i;
        a22 += q2[i] * C_q2_i;
    }

    // 2x2 Symmetric Eigendecomposition to find internal rotation angle theta
    if (fabs(a12) > 1e-6f) {
        float theta = 0.5f * atan2f(2.0f * a12, a11 - a22);
        float c = cosf(theta);
        float s = sinf(theta);

        // Rotate basis vectors to align with principal axes inside the 2D subspace
        for (int i = 0; i < dim; i++) {
            float u1 = c * q1[i] + s * q2[i];
            float u2 = -s * q1[i] + c * q2[i];
            q1[i] = u1;
            q2[i] = u2;
        }
    }

    // Step 4: Sign-locking (Prevents vectors from flipping 180 degrees)
    if (dotProduct(q1, q1_old, dim) < 0.0f) {
        for (int i = 0; i < dim; i++) q1[i] = -q1[i];
    }
    if (dotProduct(q2, q2_old, dim) < 0.0f) {
        for (int i = 0; i < dim; i++) q2[i] = -q2[i];
    }
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


int PCA(float* runningMean, float* runningCov, float* tde, float* slidingWindow, int dimensions,
        int windowSize, float newValue, int* indexes)
{
    static int sampleCount = 0;
    int tau = indexes[0] - indexes[1]; // Deduce tau spacing dynamically
    bool isWindowFull = (sampleCount >= windowSize);

    // Only extract evicted sample if the sliding window is full
    float tdeOldRaw[dimensions];
    if (isWindowFull)
    {
        for (int i = 0; i < dimensions; i++)
        {
            tdeOldRaw[i] = slidingWindow[(dimensions - 1 - i) * tau];
        }
    }

    slideWindow(slidingWindow, windowSize, newValue);
    sampleCount++;

    int minSamples = (dimensions - 1) * tau + 1;
    if (sampleCount < minSamples)
    {
        return 1; // Wait until enough delay steps have accumulated
    }

    embedding(tde, slidingWindow, dimensions, indexes); // raw, uncentered new embedded vector

    // tdeOldRaw belongs to the step BEFORE this one, so it must be centered with the mean
    // as it was before updateMean runs below, not with the just-updated mean.
    float tdeOldCentered[dimensions];

    if (isWindowFull)
    {
        centerData(runningMean, tdeOldRaw, tdeOldCentered, dimensions);
    }

    int n = isWindowFull ? (windowSize - (dimensions - 1) * tau) : (sampleCount - (dimensions - 1) * tau);

    updateMean(runningMean, dimensions, n, tde, isWindowFull ? tdeOldRaw : NULL);
    centerData(runningMean, tde, tde, dimensions);

    // Pass 'n' here to keep covariance bounded
    updateCovariance(runningCov, dimensions, tde, isWindowFull ? tdeOldCentered : NULL, n);
    return 0;
}

void updateMean(float* runningMean, int dimensions, int n, const float* newEmbedded, const float* oldEmbedded)
{
    for (int i = 0; i < dimensions; i++)
    {
        if (oldEmbedded != NULL)
        {
            // Fixed window size
            runningMean[i] += (newEmbedded[i] - oldEmbedded[i]) / (float)n;
        }
        else
        {
            // Growing window size
            runningMean[i] += (newEmbedded[i] - runningMean[i]) / (float)n;
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
                      const float* oldCentered, int n)
{
    if (n <= 1) return;

    for (int i = 0; i < dimensions; i++)
    {
        for (int j = 0; j < dimensions; j++)
        {
            int covIdx = indexAccessHelper(i, j, dimensions);
            
            if (oldCentered != NULL)
            {
                // Sliding window (fixed n): Add new, subtract old
                float contribution = (newCentered[i] * newCentered[j]) - (oldCentered[i] * oldCentered[j]);
                runningCov[covIdx] += contribution / (float)n;
            }
            else
            {
                // Growing window (increasing n): Running average update
                float contribution = newCentered[i] * newCentered[j];
                runningCov[covIdx] += (contribution - runningCov[covIdx]) / (float)n;
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