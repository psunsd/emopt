# EMopt
Refer to the documents [on readthedocs](https://emopt.readthedocs.io/en/latest/) created by Andy Michaels for details on how to install and use EMopt.  This markdown document will only contain the new APIs that are added to the GPU-EMopt.

# GPU-EMopt
In this fork, the FDTD module, the grid meshing module, and part of the gradient calculation module are ported to CUDA C++.  Although my intention at the beginning was to design the GPU-EMopt in a way such that the CPU- and GPU-modules can be swapped freely, it becomes increasingly difficult without clear benefit, so this feature will not be supported in current and future versions. 

## FDTD module
The following new parameters in the initialization of emopt.fdtd.FDTD class are exposed to users:

    gpus_count: integer; number of GPUs used for FDTD; default to 1
  
    domain_decomp: char; domain decomposition direction; options are 'x', 'y', or 'z'; default to 'x'


## Grid meshing module
The following new parameters in the initialization of emopt.grid.StructuredMaterial3D are exposed to users:

    Nsubcell: integer; number of subcells per grid cell used for fas approximate calculation of polygon clipping between geometry primitive and grid cell; default to 256
