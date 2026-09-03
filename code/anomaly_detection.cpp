#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "include/streamingData.hpp"
#include "include/AnomalyDetection.h"

void writeData(std::string filePath, std::vector<float> fileToWrite, bool append = true)
{
    std::ofstream MyFile;
    if (append)
    {
        MyFile.open(filePath, std::ios_base::app);
    }
    else
    {
        MyFile.open(filePath);
    }

    if (!MyFile.is_open())
    {
        std::cerr << "Error: Could not open or create file at path: " << filePath << std::endl;
        return;
    }

    for (std::size_t i = 0; i < fileToWrite.size(); i += 2)
    {
        MyFile << fileToWrite.at(i) << "," << fileToWrite.at(i + 1) << std::endl;
    }
    MyFile.close();
}

int main(int argc, char* argv[])
{
    std::vector<float> writeToFile;
    std::cout << "Real time anomaly detection...." << std::endl;
    //std::string fileInput = "19 - m1_mechanically_imbalanced_load_0.5Nm_m2_mechanically_imbalanced_on_background_half_speed";
    std::string fileInput = argv[4];
    std::string absolutePath = "Data/" + fileInput + ".csv";
    streamData DataStream(absolutePath);
    std::string line;

    // Time-delay embedding parameters.
    const int dimensions = std::stoi(argv[1]); // number of embedding dimensions
    int tau = std::stoi(argv[2]);        // delay between successive embedding coordinates
    const int windowSize = std::stoi(argv[3]); // length of the sliding window buffer

    std::vector<float> slidingWindow(windowSize, 0.0f); // raw streaming values
    float tde[dimensions];           // current time-delay embedding vector
    std::fill(tde, tde + dimensions, 0.0f);

    // Precompute the sliding-window offsets used to build each embedding.
    int tdeIndexes[dimensions];
    embeddingIndexes(tdeIndexes, windowSize, dimensions, tau);

    // Incrementally updated statistics for streaming PCA.
    float runningMean[dimensions] = {0.0f};
    float runningCov[dimensions * dimensions] = {0.0f};
    float runningScatter[dimensions * dimensions] = {0.0f};

    // Top two principal components (eigenvectors) of the embedding.
    float principalComponent1[dimensions] = {0.0f};
    float principalComponent2[dimensions] = {0.0f};

    // Initialize the components to the identity basis vectors.
    principalComponent1[0] = 1.0f;
    principalComponent2[1] = 1.0f;

    // Projection of the current embedding onto the two principal components.
    float outX = 0.0f;
    float outY = 0.0f;

    while (DataStream.hasNext())
    {
        DataStream.next(line);
        if (line.empty())
        {
            break;
        }
        float value = std::stof(line);

        if (processNewDataPoint(value, tde, slidingWindow.data(), runningMean, runningCov, runningScatter,principalComponent1,
                                principalComponent2, tdeIndexes, windowSize, dimensions,
                                &outX, &outY) != 1)
        {
            writeToFile.push_back(outX);
            writeToFile.push_back(outY);
        }
    }
    std::string outputPath = "output/output.txt";
    writeData(outputPath, writeToFile, false);
    return 0;
}