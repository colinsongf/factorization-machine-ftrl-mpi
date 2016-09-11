#!/bin/bash
GLOG_LIB = -L/usr/local/lib
GLOG_INCLUDE = -I/usr/local/include/glog
GTEST_LIB = -L/usr/local/lib
GTEST_INCLUDE = -I/usr/local/include/gtest
#train code
CPP_tag = -std=gnu++11

LIB=/home/services/xiaoshu/lib
INCLUDE=/home/services/xiaoshu/include
#train code
train:main.o
	mpicxx $(CPP_tag) -o train main.o -lpthread

main.o: src/main.cpp 
	mpicxx $(CPP_tag) -c src/main.cpp

clean:
	rm -f *~ train predict *.o
