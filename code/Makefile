# Simple Makefile for Windows with MinGW

# Compilers
CXX = g++
CC = gcc

# Flags
# -static-libgcc/-static-libstdc++: this machine has more than one MinGW toolchain on PATH
# (Strawberry Perl's and Git's bundled one). Dynamically linking libstdc++-6.dll risks
# loading a different toolchain's copy at runtime than the one used to compile, which is
# an ABI mismatch that crashes inside the C++ standard library (seen in std::ifstream).
# Static-linking removes the dependency on which DLL happens to be found first in PATH.
CXXFLAGS = -std=c++17 -O2 -Wall -Iinclude -Ieigen-3.4.0 -static-libgcc -static-libstdc++
CFLAGS = -std=c11 -O2 -Wall -Iinclude

# Directories
SRC_DIR = src
INCLUDE_DIR = include
OUTPUT_DIR = output

# Target name
TARGET = anomaly_detection

# Find all C++ source files
CXX_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CXX_SOURCES += anomaly_detection.cpp

# Find all C source files
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)

# Generate object file names
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# All objects
OBJECTS = $(CXX_OBJECTS) $(C_OBJECTS)

# Default target
all: $(TARGET)

# Build the program by linking all object files
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET).exe $(OBJECTS)
	@echo "Build complete: $(TARGET).exe"

# Compile C++ source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Alternative: single command to compile everything
quick:
	$(CXX) $(CXXFLAGS) -o $(TARGET).exe $(CXX_SOURCES) $(C_SOURCES)

# Run the program
run: $(TARGET)
	./$(TARGET).exe

# Clean the executable and object files
clean:
	del $(TARGET).exe $(SRC_DIR)\*.o *.o 2>nul || true

# Clean everything including object files
cleanall: clean
	@echo "Cleaned all build files"

# Help command
help:
	@echo "Commands (use mingw32-make):"
	@echo "  mingw32-make       - Build the program"
	@echo "  mingw32-make quick - Quick build (single command)"
	@echo "  mingw32-make run   - Build and run"
	@echo "  mingw32-make clean - Remove executable and object files"

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: CFLAGS += -g -DDEBUG
debug: clean $(TARGET)
	@echo "Debug build complete"

.PHONY: all run clean cleanall help debug quick