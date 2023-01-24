FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

USER root
WORKDIR /home
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV LC_ALL "C.UTF-8"
ENV HOME "/home"
ENV PATH "$PATH:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

RUN apt-get update && apt-get -y install build-essential gfortran wget
RUN cd /home && wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz && tar xvzf openmpi-4.0.3.tar.gz 
RUN cd /home/openmpi-4.0.3 && ./configure --prefix=/home/opt/openmpi && make all && make install

ENV PATH "$PATH:/home/opt/openmpi/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/home/opt/openmpi/lib"
ENV OMPI_ALLOW_RUN_AS_ROOT 1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1

RUN echo $PATH && echo $LD_LIBRARY_PATH
RUN apt-get -y install git python3.6 python3.6-dev python3-pip python-tk
RUN pip3 install requests h5py==3.1.0 kiwisolver==1.0.1 matplotlib==3.1.3 numpy==1.18.1 scipy==1.4.1 --user

RUN cd /home && git clone https://github.com/psunsd/emopt.git && cd emopt && python3.6 install.py 
ENV PETSC_DIR="/home/.emopt"
ENV SLEPC_DIR="/home/.emopt"
RUN pip3 install mpi4py==3.0.3 petsc4py==3.12.0 slepc4py==3.12.0 future --user
RUN cd /home/emopt/ && python3.6 setup.py install --user

RUN cd /home/emopt/src && nvcc -Xcompiler -fPIC -O3 -std=c++14 -arch sm_52 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 -shared fdtd.cu -o FDTD.so && \
	nvcc -Xcompiler -fPIC -c -arch sm_52 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 Grid_CUDA.cu && \
	g++ -c -fPIC Grid_CPP.cpp -fopenmp -O3 -march=native -DNDEBUG -std=c++14 -o Grid.o -I/home/.emopt/include/ -I/usr/local/cuda/include/ && \
	g++ -shared -fopenmp -fPIC -o Grid.so Grid.o Grid_CUDA.o -lpthread -lrt -ldl -L/usr/local/cuda/lib64 -lcudart_static -lculibos && \
	mv *.so /home/.local/lib/python3.6/site-packages/emopt-2023.1.16-py3.6.egg/emopt/

ENV PYTHONPATH "/home/.local/lib/python3.6/site-packages:/home/.local/lib/python3.6/site-packages/emopt-2023.1.16-py3.6.egg"
