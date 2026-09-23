CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
TARGETS = gen_dataset

.PHONY: all clean

all: $(TARGETS)

gen_dataset: gen_dataset.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o $(TARGETS)
