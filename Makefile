TMP=$(shell which nvcc)
CUDA_INCLUDE=$(shell echo ${TMP} | sed "s/\/bin\/nvcc/\/include\//g")
SOURCES = $(wildcard *.cpp *.cu)

all: Makefile main.cpp
	nvcc -std=c++11 -I. -I${CUDA_INCLUDE} ${SOURCES} -o main -L/usr/X11R6/lib -lm  -lpthread -lX11  `pkg-config --libs opencv` `pkg-config --cflags opencv`

