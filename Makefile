TMP=$(shell which nvcc)
CUDA_INCLUDE=$(shell echo ${TMP} | sed "s/\/bin\/nvcc/\/include\//g")
LIBTIFF_INCLIDE=/misc/lmbraid12/tananaev/libtiff/libtiff-3.0.0/libtiff
SOURCES = $(wildcard *.cpp *.cu)

all: Makefile main.cpp
	nvcc -g -G -lineinfo -std=c++11 -I. -I${CUDA_INCLUDE} ${SOURCES} -o main -L/usr/X11R6/lib -lm  -lpthread -lX11 -L/misc/lmbraid12/tananaev/lib -ltiff

optim: Makefile main.cpp
	nvcc -std=c++11 -I. -I${CUDA_INCLUDE} ${SOURCES} -o main -L/usr/X11R6/lib -lm  -lpthread -lX11 -O3 -L/misc/lmbraid12/tananaev/lib/libtiff -ltiff
