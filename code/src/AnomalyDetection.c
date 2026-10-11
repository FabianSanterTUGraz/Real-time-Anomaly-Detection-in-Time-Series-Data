#include "../include/AnomalyDetection.h"

// Main api call
static int counter = 0;
int processNewDataPoint(double newValue, double* tde, double* slidingWindow, double* runningMean,
                        double* runningCov, double* runningScatter,double* principalComponent1, double* principalComponent2, int* indexes,
                        int windowSize, int dimensions, double* outX, double* outY)
{
    if (PCA(runningMean, runningCov, runningScatter,tde, slidingWindow, dimensions, windowSize, newValue, indexes) == 1)
    {
        return 1;
    }

    if(counter > 200)
    {
        counter = 0;
        subspaceIteration(runningCov, dimensions, principalComponent1, principalComponent2);
    }
    counter++;

    *outX = dotProduct(tde,principalComponent1,dimensions);
    *outY = dotProduct(tde,principalComponent2,dimensions);

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

void embeddingIndexes(int* buffer, int windowSize, int dimensions, int tau)
{
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
q2: Point along the second longest axis while staying strictly 90 degree to the fist component.
*/

void subspaceIteration(const double* runningCov, int dim, double* q1, double* q2)
{
    //for later comparison between the original directions and the resulting point
    //(to ensure that the orientation is locked)
    double q1_old[dim];
    double q2_old[dim];

    copyArray(q1,q1_old,dim);
    copyArray(q2,q2_old,dim);

    // Z = C * Q
    //if a random vector is multiplied by C the matrix rotates and stretches toward
    // the direction of maximum variance (largest eigenvector)
    double z1[dim];
    double z2[dim];

    for (int i = 0; i < dim; i++)
    {
        z1[i] = 0.0f;
        z2[i] = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            double C = runningCov[indexAccessHelper(i,j,dim)];
            z1[i] += C * q1[j];
            z2[i] += C * q2[j];
        }
    }
    //after the stretching and rotating the length is not equal to 1 anymore
    // additionaly c pulls everything toward the first principle component

    //1.0 Normalize the first pc by dividing z1 by its length
    double norm1 = sqrtf(dotProduct(z1, z1, dim));
    if (norm1 > 1e-5f)
    {
        for (int i = 0; i < dim; i++)
        {
            q1[i] = z1[i] / norm1;
        }
    }

    //force z2 to be 90 degree onto z1
    double proj = dotProduct(q1, z2, dim);
    for (int i = 0; i < dim; i++) z2[i] -= proj * q1[i];

    //normalization of q2
    double norm2 = sqrtf(dotProduct(z2, z2, dim));
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

int PCA(double* runningMean, double* runningCov, double* runningScatter,double* tde, double* slidingWindow, int dimensions,
        int windowSize, double newValue, int* indexes)
{
    static int sampleCount = 0; // reusing in C++/Java frage wichtig static

    int tau = indexes[0] - indexes[1];
    bool isWindowFull = (sampleCount >= windowSize);

    double tdeOldRaw[dimensions];
    double tdeOldCentered[dimensions];
    double* oldCenteredAddress = NULL;
    double* oldRawAddress = NULL;

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

    double tdeCenteredOldMean[dimensions];
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

void updateMean(double* runningMean, int dimensions, int sampleSize, const double* newEmbedded, const double* oldEmbedded)
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

void updateCovariance(double* runningCov, int dimensions, const double* newCentered,
                      const double* oldCentered, int sampleSize)
{
    if (sampleSize <= 1) return;

    for (int i = 0; i < dimensions; i++)
    {
        for (int j = 0; j < dimensions; j++)
        {
            int covIdx = indexAccessHelper(i, j, dimensions);
            double contribution = (newCentered[i] * newCentered[j]) - (oldCentered[i] * oldCentered[j]);
            runningCov[covIdx] += contribution / (double)sampleSize;
        }
    }
}

void updateCovarianceIncremental(double* runningScatter, int dimensions,
                                  const double* deltaOld, const double* deltaNew, int sampleSize)
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

void scatterToCovariance(const double* scatter, double* covOut, int dimensions, int sampleSize)
{
    double denom = (sampleSize > 1) ? (double)(sampleSize - 1) : 1.0f;
    for (int k = 0; k < dimensions * dimensions; k++)
    {
        covOut[k] = scatter[k] / denom;
    }
}

double dotProduct(const double* v1, const double* v2, int dim)
{
    double sum = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        sum += v1[i] * v2[i];
    }
    return sum;
}