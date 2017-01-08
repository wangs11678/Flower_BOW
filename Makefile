INCDIR = -I.
DBG    = -g
OPT    = -O3
CPP    = g++
CFLAGS = $(DBG) $(OPT) $(INCDIR)
LINK   = -lm 

CXXFLAGS = $(shell pkg-config --cflags opencv)
LDLIBS = $(shell pkg-config --libs opencv)

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@


bow: sift_bow_svm.cpp
	$(CPP) $(CFLAGS) -o bow sift_bow_svm.cpp $(LINK) $(CXXFLAGS) $(LDLIBS)

clean:
	/bin/rm -f bow *.o

clean-all: clean
	/bin/rm -f *~ 



