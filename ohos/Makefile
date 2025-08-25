# Simple Makefile for testing the OpenHarmony C++ adaptation
# For production builds, use CMake

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I./include
LIBS = -lm

SRCDIR = src
DEMODIR = demo
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
DEMO_SOURCES = $(wildcard $(DEMODIR)/*.cpp)

LIBRARY = libqwen_inference.a
DEMO = qwen_demo

.PHONY: all clean test library demo

all: library demo

library: $(LIBRARY)

demo: $(DEMO)

$(LIBRARY): $(OBJECTS)
	ar rcs $@ $^

$(DEMO): $(LIBRARY) $(DEMO_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEMO_SOURCES) $(LIBRARY) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test: $(DEMO)
	@echo "Build successful! Demo executable created: $(DEMO)"
	@echo "Note: Demo requires model files to run"

clean:
	rm -f $(OBJECTS) $(LIBRARY) $(DEMO)
	rm -f *.o

# Show the structure
structure:
	@echo "Project structure:"
	@find . -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.txt" -o -name "*.md" | grep -v "build" | sort