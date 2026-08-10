#include <fstream>
#include <iostream>
#include <vector>

#include "include/streamingData.hpp"
#include "include/AnomalyDetection.h"
#include "include/utils.h"

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

int main()
{
    std::vector<float> writeToFile;
    std::cout << "Real time anomaly detection...." << std::endl;
    int i = 20;
    std::string fileInput = std::to_string(i);
    std::string absolutePath = "Data/" + fileInput + ".csv";
    streamData DataStream(absolutePath);
    std::string line;

    // Settings of sliding window and tde vector.
    // dimensions/tau must match d/tau in static_fingerprintvisualization.py
    // so the streaming and batch PCA embed the same windows for computeDelta.py.
    int dimensions = 13;
    int tau = 15;
    int windowSize = 111562; // theoretisch extrem groß wählen

    float slidingWindow[windowSize] = {0.0f};
    float tde[dimensions] = {0.0f};

    int tdeIndexes[dimensions];
    embeddingIndexes(tdeIndexes, windowSize, dimensions, tau);

    float runningMean[dimensions] = {0.0f};
    float runningCov[dimensions * dimensions] = {0.0f};

    float principalComponent1[dimensions] = {0.0f};
    float principalComponent2[dimensions] = {0.0f};
    
    principalComponent1[0] = 1.0f; // Vprev starts at identity
    principalComponent2[1] = 1.0f; // Vprev starts at identity


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

        if (processNewDataPoint(value, tde, slidingWindow, runningMean, runningCov, principalComponent1,
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