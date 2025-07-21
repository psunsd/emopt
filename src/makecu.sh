#!/bin/bash

echo "Compiling FDTD..."
nvcc -Xcompiler -fPIC -O3 -std=c++14 -arch sm_52 \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 \
	-gencode=arch=compute_87,code=sm_87 \
	-gencode=arch=compute_90,code=sm_90 \
	-gencode=arch=compute_86,code=compute_86 -shared fdtd.cu -o FDTD.so

echo "Compiling Grid..."
nvcc -Xcompiler -fPIC -c -arch sm_52 \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 \
	-gencode=arch=compute_87,code=sm_87 \
	-gencode=arch=compute_90,code=sm_90 \
	-gencode=arch=compute_86,code=compute_86 Grid_CUDA.cu -o Grid_CUDA.o
g++ -c -fPIC Grid_CPP.cpp -fopenmp -O3 -march=x86-64 -DNDEBUG -std=c++14 -o Grid.o -I/home/.emopt/include/ -I/usr/local/cuda/include/
echo "Linking Grid..."
g++ -shared -fopenmp -fPIC -o Grid.so Grid.o Grid_CUDA.o -lpthread -lrt -ldl -L/usr/local/cuda/lib64 -lcudart_static -lculibos
echo "Copying objects..."
declare SITEPKGINFO=($(pip3 show emopt))
EMOPTPATH=${SITEPKGINFO[28]}
EMOPTPATH+="/emopt/"
#mv *.so /home/.local/lib/python3.6/site-packages/emopt-2023.1.16-py3.6.egg/emopt/
mv *.so $EMOPTPATH
