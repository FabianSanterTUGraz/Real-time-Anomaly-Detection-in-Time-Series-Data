# Makefile for Windows (MinGW / MSYS2) and Linux
#
#   make                 release build -> anomaly_detection(.exe)        (-O2)
#   make debug           debug build   -> anomaly_detection_debug(.exe)  (-O0 -g)
#   make run             build + run with ARGS (default: 600 1 20480 input)
#   make run ARGS="100 1 2560 input"
#   make validation      build validation(.exe) from validation.cpp + C core
#   make clean           remove build/ and all executables
#
# Objects go to build/release or build/debug, so release and debug never mix and the
# source folders stay clean. Header dependencies are tracked (-MMD -MP): changing
# include/AnomalyDetection.h rebuilds everything that includes it, so no stale .o files.

# ---------------------------------------------------------------- OS / shell detection
# mingw32-make runs recipes with sh.exe only if it finds one on PATH (e.g. inside the MSYS2
# terminal); from PowerShell, cmd or CLion it uses cmd.exe. To behave the same everywhere,
# mingw32-make is forced to cmd.exe and all file commands come in a cmd and a POSIX variant.
ifeq ($(OS),Windows_NT)
    EXE_EXT  = .exe
    MAKE_CMD = mingw32-make
    # More than one MinGW toolchain is on PATH (Strawberry Perl's and Git's bundled one).
    # Dynamically linking libstdc++-6.dll risks loading another toolchain's copy at runtime
    # (ABI mismatch, crashed inside std::ifstream). Static linking removes that dependency.
    PLATFORM_LDFLAGS = -static-libgcc -static-libstdc++
else
    EXE_EXT  =
    MAKE_CMD = make
    PLATFORM_LDFLAGS =
endif

ifneq ($(findstring mingw32-make,$(MAKE)),)
    SHELL := cmd.exe
    winpath = $(subst /,\,$(1))
    MKDIR_P = if not exist "$(call winpath,$(1))" mkdir "$(call winpath,$(1))"
    RM_DIR  = if exist "$(call winpath,$(1))" rmdir /s /q "$(call winpath,$(1))"
    RM_FILES = del /q /f $(call winpath,$(1)) 2>nul
    # a trailing backslash would continue the line in make, hence the empty $(strip)
    RUN_PREFIX = .\$(strip)
else
    MKDIR_P = mkdir -p $(1)
    RM_DIR  = rm -rf $(1)
    RM_FILES = rm -f $(1)
    RUN_PREFIX = ./
endif

# ---------------------------------------------------------------- tools and flags
CC  = gcc
CXX = g++

INCLUDE_DIR = include
SRC_DIR     = src

WARN     = -Wall
# -MMD -MP: write .d files with header dependencies next to each object
CPPFLAGS = -I$(INCLUDE_DIR) -MMD -MP
CFLAGS   = -std=c11 $(WARN)
CXXFLAGS = -std=c++17 $(WARN)
LDFLAGS  = $(PLATFORM_LDFLAGS)
# libm for sqrt etc.: required on Linux, harmless on MinGW
LDLIBS   = -lm

ifeq ($(DEBUG),1)
    MODE      = debug
    OPT       = -O0 -g -DDEBUG
    EXE_SUFFIX = _debug
else
    MODE      = release
    OPT       = -O2
    EXE_SUFFIX =
endif
BUILD_DIR = build/$(MODE)

# ---------------------------------------------------------------- sources and targets
TARGET            = anomaly_detection$(EXE_SUFFIX)$(EXE_EXT)
VALIDATION_TARGET = validation$(EXE_SUFFIX)$(EXE_EXT)

# portable C core / C++ helpers (e.g. streamingData)
C_SOURCES   = $(wildcard $(SRC_DIR)/*.c)
CXX_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)

C_OBJECTS   = $(C_SOURCES:%.c=$(BUILD_DIR)/%.o)
CXX_OBJECTS = $(CXX_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
MAIN_OBJECT = $(BUILD_DIR)/anomaly_detection.o
VALIDATION_OBJECT = $(BUILD_DIR)/validation.o

ALL_OBJECTS = $(C_OBJECTS) $(CXX_OBJECTS) $(MAIN_OBJECT) $(VALIDATION_OBJECT)
DEPS = $(ALL_OBJECTS:.o=.d)

ARGS ?= 600 1 20480 input

# ---------------------------------------------------------------- rules
all: $(TARGET)

$(TARGET): $(MAIN_OBJECT) $(CXX_OBJECTS) $(C_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	@echo Build complete: $@ - $(MODE)

$(VALIDATION_TARGET): $(VALIDATION_OBJECT) $(CXX_OBJECTS) $(C_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	@echo Build complete: $@ - $(MODE)

# C files are compiled as C (gcc), C++ files as C++ (g++)
$(BUILD_DIR)/%.o: %.c
	-@$(call MKDIR_P,$(patsubst %/,%,$(dir $@)))
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPT) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	-@$(call MKDIR_P,$(patsubst %/,%,$(dir $@)))
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPT) -c $< -o $@

-include $(DEPS)

# ---------------------------------------------------------------- convenience targets
debug:
	@$(MAKE) --no-print-directory DEBUG=1 all

validation: $(VALIDATION_TARGET)

run: $(TARGET)
	$(RUN_PREFIX)$(TARGET) $(ARGS)

run-validate: $(VALIDATION_TARGET)
	$(RUN_PREFIX)$(VALIDATION_TARGET)

clean:
	-@$(call RM_DIR,build)
	-@$(call RM_FILES,anomaly_detection$(EXE_EXT) anomaly_detection_debug$(EXE_EXT) validation$(EXE_EXT) validation_debug$(EXE_EXT) *.o $(SRC_DIR)/*.o)
	@echo Cleaned

# no quotes or parentheses in the help text: cmd prints quotes literally, sh chokes on ( )
help:
	@echo Commands - use $(MAKE_CMD):
	@echo   $(MAKE_CMD)                 release build, -O2: $(TARGET)
	@echo   $(MAKE_CMD) debug           debug build, -O0 -g: anomaly_detection_debug$(EXE_EXT)
	@echo   $(MAKE_CMD) run             build and run with ARGS=$(ARGS)
	@echo   $(MAKE_CMD) run ARGS=...    other arguments: d tau windowSize inputName
	@echo   $(MAKE_CMD) validation      build the validation program
	@echo   $(MAKE_CMD) run-validate    build and run the validation program
	@echo   $(MAKE_CMD) clean           remove build folder and executables

.PHONY: all debug validation run run-validate clean help