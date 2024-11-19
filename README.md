# EMopt
Refer to the documents [on readthedocs](https://emopt.readthedocs.io/en/latest/) created by Andy Michaels for details on how to install and use EMopt.  This markdown document will only contain new APIs of the GPU-EMopt.

# GPU-EMopt
The **FDTD** module, the **grid meshing** module, and part of the **gradient calculation** module are ported to CUDA C++.  To enable multi-GPU FDTD, the original FDTD control flow in fdtd.py is absorbed in C++, but the APIs exposed to users are not changed.

## Docker images hosted on dockerhub
Original CPU version:

    docker pull psunsd/emopt:2023.1.16
CUDA 12.2.0 + amd64: 

    docker pull psunsd/emopt-gpu:cuda12.2.0
CUDA 12.2.2 + arm64: 
    
    docker pull psunsd/emopt-gpu-arm64:cuda12.2.0

## FDTD module
The GPU-FDTD module is benchmarked on 13 GPU models of Volta, Turing, Ampere and Hopper architectures.  A single Tesla V100 GPU can achieve ~3x FDTD throughput as that of the CPU-EMopt on an HPE Superdome Flex S280 server with 16x 18-core CPUs.

![image](https://github.com/psunsd/emopt/assets/61566314/436c1790-d0fe-4a05-8592-eef22500a4de)

Parallel efficiency of the GPU-FDTD module is benchmarked on a DGX-2 with 16x Tesla V100 SXM3, which are connected by all-to-all NVLink/NVSwitch fabrics.  Each GPU requires >4M FDTD nodes to effectively hide latency.  If a small simulation is distributed among too many GPUs, the latency will emerge and the paralle efficiency will drop.

![image](https://github.com/psunsd/emopt/assets/61566314/26cc6c93-0beb-4b89-bf5b-ae0f431ddba1)

The following new parameters in the initialization of emopt.fdtd.FDTD class are exposed to users:

    gpus_count: integer; number of GPUs used for FDTD; default to 1
  
    domain_decomp: char; domain decomposition direction; options are 'x', 'y', or 'z'; default to 'x'

## Grid meshing module
The following new parameters in the initialization of emopt.grid.StructuredMaterial3D are exposed to users:

    Nsubcell: integer; number of subcells per grid cell used for fas approximate calculation of polygon clipping between geometry primitive and grid cell; default to 256

The FDTD and the grid meshing modules are profiled using the NVIDIA Nsight Compute tool: both modules are compute-bound, and have achieved ~85% of GPU's peak performance.

![image](https://github.com/psunsd/emopt/assets/61566314/b9af4f52-8499-442c-98ce-84597ffd3e95)

