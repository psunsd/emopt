FROM nvidia/cuda:12.2.2-devel-ubuntu20.04

USER root
WORKDIR /home
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV LC_ALL "C.UTF-8"
ENV HOME "/home"
ENV PATH "$PATH:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

RUN apt-get update && apt-get -y install build-essential gfortran wget
RUN echo $PATH && echo $LD_LIBRARY_PATH
RUN apt-get -y install git python3.8 python3.8-dev python3-pip python-tk python-is-python3
RUN pip3 install requests 
RUN apt install -y python3-h5py python3-matplotlib python3-numpy python3-scipy
RUN pip3 install kiwisolver==1.0.1 --user

RUN cd /home && wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz && tar xvzf openmpi-4.0.3.tar.gz 
RUN cd /home/openmpi-4.0.3 && ./configure --prefix=/home/opt/openmpi && make all && make install

ENV PATH "$PATH:/home/opt/openmpi/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/home/opt/openmpi/lib"
ENV OMPI_ALLOW_RUN_AS_ROOT 1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1

RUN pip3 install numpy==1.23.5 --user
RUN cd /home && git clone https://github.com/psunsd/emopt.git && cd emopt && python3.8 install_arm64.py 
ENV PETSC_DIR="/home/.emopt"
ENV SLEPC_DIR="/home/.emopt"
RUN pip3 install mpi4py==3.0.3 --user
RUN pip3 install petsc4py==3.16.0 --user
RUN pip3 install slepc4py==3.16.0 --user
RUN pip3 install future --user
RUN cd /home/emopt/ && python3.8 setup.py install --user
RUN pip3 install shapely --user

RUN apt-get install -y bash
RUN cd /home/emopt/src && bash makecu.sh

ENV PYTHONPATH "/home/.local/lib/python3.8/site-packages:/home/.local/lib/python3.8/site-packages/emopt-2023.1.16-py3.8.egg"

ARG TARGETARCH
RUN echo "Building for architecture: ${TARGETARCH}"
RUN if [ "${TARGETARCH}" = "amd64" ]; then\
        pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.4.1+cu121 networkx==2.8.8
    elif [ "${TARGETARCH}" = "arm64" ]; then \
        pip3 install torch==2.4.1 networkx==2.8.8; \
    else \
        echo "Unsupported architecture: ${TARGETARCH}" && exit 1; \
    fi
RUN sed -i '153i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ self._update_fun()' /usr/lib/python3/dist-packages/scipy/optimize/_differentiable_functions.py
RUN sed -i '154i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ self._update_grad()' /usr/lib/python3/dist-packages/scipy/optimize/_differentiable_functions.py
