#include <complex>
#include <thrust/complex.h>

typedef struct kernelparams {
    int i1, i2, j1, j2, k1, k2, Nx, Ny, Nz;
    int NcellBlk;
    double dx, dy, dz;
    double sx, sy, sz;
    double background;
    size_t size;
    int PrimLen, LayerLen, ZLen, VerticesLen;
    thrust::complex<double> *grid, *PrimMatValue;
    double *ZList, *PrimVertXAll, *PrimVertYAll, *PrimVertXmin, *PrimVertXmax, *PrimVertYmin, *PrimVertYmax;
    int *PrimLayerSeq, *PrimSeqWithinLayer, *PrimVertCoordLoc, *PrimVertNumber, *PrimNumberPerLayer, *PrimIdxPerLayer, *PrimLayerValue;

    int Nsubcell;
    bool *subcellflag;
    double *cellarea, *overlap;
    int *inpoly;
} kernelpar;

void invoke_kernel_get_values_Ncell(kernelpar *kpar, dim3 grid_dim, dim3 block_dim);