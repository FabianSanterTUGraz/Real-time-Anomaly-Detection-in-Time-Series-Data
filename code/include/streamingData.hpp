#ifndef STREAMINGDATA_HPP
#define STREAMINGDATA_HPP

#include <fstream>
#include <iostream>

class streamData
{
  private:
    std::ifstream Stream;

  public:
    streamData(const std::string& filePath) : Stream(filePath)
    {
        std::string header;
        std::getline(Stream, header);
    };

    bool next(std::string& line)
    {
        return static_cast<bool>(std::getline(Stream, line));
    };

    bool hasNext()
    {
        if (Stream.peek() != EOF)
        {
            return true;
        }
        return false;
    }

    void readData();
};

#endif