#include "fdtd.hpp"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <thrust/complex.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__global__ void kernel_update_H(kernelpar *kpar)
{
//     1D grid of 1D blocks
    size_t ind_global = blockIdx.x * blockDim.x + threadIdx.x;
//     1D grid of 2D blocks
//     size_t ind_global = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
//     2D grid of 1D blocks
//     size_t ind_global = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
    if (ind_global >= kpar->size){
        return;
    }
    int i, j, k, ind_ijk, ind_ijp1k, ind_ip1jk, ind_ijkp1;
    double dExdy, dExdz, dEydx, dEydz, dEzdx, dEzdy;
    int ind_pml, ind_pml_param;
    double b, C, kappa;
    int //srcind_ijk,
        srcind_src, srcind_size, srcind_size_offset;
    double src_t;

    // ind_global = i*_J*_K + j*_K + k;
    i = ind_global/(kpar->J*kpar->K);
    j = (ind_global%(kpar->J*kpar->K))/kpar->K;
    k = (ind_global%(kpar->J*kpar->K))%kpar->K;

    //blockDim.x = 128; gridDim.x=585
//     k = ind_global/(kpar->I*kpar->J);
//     j = (ind_global%(kpar->I*kpar->J))/kpar->I;
//     i = (ind_global%(kpar->I*kpar->J))%kpar->I;

    ind_ijk = (i+1)*(kpar->J+2)*(kpar->K+2) + (j+1)*(kpar->K+2) + k + 1;
    ind_ijp1k = ind_ijk + kpar->K + 2;
    ind_ip1jk = ind_ijk + (kpar->J+2)*(kpar->K+2);
    ind_ijkp1 = ind_ijk + 1;

    // set up fields on the boundary
    if (kpar->bc[0] != 'P' && k == kpar->Nx - kpar->k0){
        kpar->Ey[ind_ijk] = 0.0;
        kpar->Ez[ind_ijk] = 0.0;
    }
    if (kpar->bc[1] != 'P' && j == kpar->Ny - kpar->j0){
        kpar->Ex[ind_ijk] = 0.0;
        kpar->Ez[ind_ijk] = 0.0;
    }
    if (kpar->bc[2] != 'P' && i == kpar->Nz - kpar->i0){
        kpar->Ex[ind_ijk] = 0.0;
        kpar->Ey[ind_ijk] = 0.0;
    }

    dEzdy = kpar->odx * (kpar->Ez[ind_ijp1k] - kpar->Ez[ind_ijk]);
    dEydz = kpar->odx * (kpar->Ey[ind_ip1jk] - kpar->Ey[ind_ijk]);
    dExdz = kpar->odx * (kpar->Ex[ind_ip1jk] - kpar->Ex[ind_ijk]);
    dEzdx = kpar->odx * (kpar->Ez[ind_ijkp1] - kpar->Ez[ind_ijk]);
    dEydx = kpar->odx * (kpar->Ey[ind_ijkp1] - kpar->Ey[ind_ijk]);
    dExdy = kpar->odx * (kpar->Ex[ind_ijp1k] - kpar->Ex[ind_ijk]);
    kpar->Hx[ind_ijk] = kpar->Hx[ind_ijk] + kpar->dt * (dEydz - dEzdy);
    kpar->Hy[ind_ijk] = kpar->Hy[ind_ijk] + kpar->dt * (dEzdx - dExdz);
    kpar->Hz[ind_ijk] = kpar->Hz[ind_ijk] + kpar->dt * (dExdy - dEydx);

    // Update PML
    if (k + kpar->k0 < kpar->pml_xmin){
        ind_pml = i * kpar->J * (kpar->pml_xmin - kpar->k0) + j * (kpar->pml_xmin - kpar->k0) + k;
        ind_pml_param = kpar->pml_xmin - k - kpar->k0 - 1;
        kappa = kpar->kappa_H_x[ind_pml_param];
        b = kpar->bHx[ind_pml_param];
        C = kpar->cHx[ind_pml_param];
        kpar->pml_Eyx0[ind_pml] = C * dEydx + b * kpar->pml_Eyx0[ind_pml];
        kpar->pml_Ezx0[ind_pml] = C * dEzdx + b * kpar->pml_Ezx0[ind_pml];
        kpar->Hz[ind_ijk] = kpar->Hz[ind_ijk] - kpar->dt * (kpar->pml_Eyx0[ind_pml] - dEydx + dEydx / kappa);
        kpar->Hy[ind_ijk] = kpar->Hy[ind_ijk] + kpar->dt * (kpar->pml_Ezx0[ind_pml] - dEzdx + dEzdx / kappa);
    }
    else if(k + kpar->k0 >= kpar->pml_xmax){
        ind_pml = i * kpar->J * (kpar->k0 + kpar->K - kpar->pml_xmax) + j * (kpar->k0 + kpar->K - kpar->pml_xmax)
                + k + kpar->k0 - kpar->pml_xmax;
        ind_pml_param = k + kpar->k0 - kpar->pml_xmax + kpar->w_pml_x0;
        kappa = kpar->kappa_H_x[ind_pml_param];
        b = kpar->bHx[ind_pml_param];
        C = kpar->cHx[ind_pml_param];
        kpar->pml_Eyx1[ind_pml] = C * dEydx + b * kpar->pml_Eyx1[ind_pml];
        kpar->pml_Ezx1[ind_pml] = C * dEzdx + b * kpar->pml_Ezx1[ind_pml];
        kpar->Hz[ind_ijk] = kpar->Hz[ind_ijk] - kpar->dt * (kpar->pml_Eyx1[ind_pml] - dEydx + dEydx / kappa);
        kpar->Hy[ind_ijk] = kpar->Hy[ind_ijk] + kpar->dt * (kpar->pml_Ezx1[ind_pml] - dEzdx + dEzdx / kappa);
    }

    if (j + kpar->j0 < kpar->pml_ymin){
        ind_pml = i * (kpar->pml_ymin - kpar->j0) * kpar->K + j * kpar->K + k;
        ind_pml_param = kpar->pml_ymin - j - kpar->j0 - 1;
        kappa = kpar->kappa_H_y[ind_pml_param];
        b = kpar->bHy[ind_pml_param];
        C = kpar->cHy[ind_pml_param];
        kpar->pml_Exy0[ind_pml] = C * dExdy + b * kpar->pml_Exy0[ind_pml];
        kpar->pml_Ezy0[ind_pml] = C * dEzdy + b * kpar->pml_Ezy0[ind_pml];
        kpar->Hz[ind_ijk] = kpar->Hz[ind_ijk] + kpar->dt * (kpar->pml_Exy0[ind_pml] - dExdy + dExdy / kappa);
        kpar->Hx[ind_ijk] = kpar->Hx[ind_ijk] - kpar->dt * (kpar->pml_Ezy0[ind_pml] - dEzdy + dEzdy / kappa);
    }
    else if(j + kpar->j0 >= kpar->pml_ymax){
        ind_pml = i * (kpar->j0 + kpar->J - kpar->pml_ymax) * kpar->K + (kpar->j0 + j - kpar->pml_ymax) * kpar->K + k;
        ind_pml_param = j + kpar->j0 - kpar->pml_ymax + kpar->w_pml_y0;
        kappa = kpar->kappa_H_y[ind_pml_param];
        b = kpar->bHy[ind_pml_param];
        C = kpar->cHy[ind_pml_param];
        kpar->pml_Exy1[ind_pml] = C * dExdy + b * kpar->pml_Exy1[ind_pml];
        kpar->pml_Ezy1[ind_pml] = C * dEzdy + b * kpar->pml_Ezy1[ind_pml];
        kpar->Hz[ind_ijk] = kpar->Hz[ind_ijk] + kpar->dt * (kpar->pml_Exy1[ind_pml] - dExdy + dExdy / kappa);
        kpar->Hx[ind_ijk] = kpar->Hx[ind_ijk] - kpar->dt * (kpar->pml_Ezy1[ind_pml] - dEzdy + dEzdy / kappa);
    }

    if (i + kpar->i0 < kpar->pml_zmin){
        ind_pml = i * kpar->J * kpar->K + j * kpar->K + k;
        ind_pml_param = kpar->pml_zmin - i - kpar->i0 - 1;
        kappa = kpar->kappa_H_z[ind_pml_param];
        b = kpar->bHz[ind_pml_param];
        C = kpar->cHz[ind_pml_param];
        kpar->pml_Exz0[ind_pml] = C * dExdz + b * kpar->pml_Exz0[ind_pml];
        kpar->pml_Eyz0[ind_pml] = C * dEydz + b * kpar->pml_Eyz0[ind_pml];
        kpar->Hx[ind_ijk] = kpar->Hx[ind_ijk] + kpar->dt * (kpar->pml_Eyz0[ind_pml] - dEydz + dEydz / kappa);
        kpar->Hy[ind_ijk] = kpar->Hy[ind_ijk] - kpar->dt * (kpar->pml_Exz0[ind_pml] - dExdz + dExdz / kappa);
    }
    else if(i + kpar->i0 > kpar->pml_zmax){
        ind_pml = (kpar->i0 + i - kpar->pml_zmax) * kpar->J * kpar->K + j * kpar->K + k;
        ind_pml_param = i + kpar->i0 - kpar->pml_zmax + kpar->w_pml_z0;
        kappa = kpar->kappa_H_z[ind_pml_param];
        b = kpar->bHz[ind_pml_param];
        C = kpar->cHz[ind_pml_param];
        kpar->pml_Exz1[ind_pml] = C * dExdz + b * kpar->pml_Exz1[ind_pml];
        kpar->pml_Eyz1[ind_pml] = C * dEydz + b * kpar->pml_Eyz1[ind_pml];
        kpar->Hx[ind_ijk] = kpar->Hx[ind_ijk] + kpar->dt * (kpar->pml_Eyz1[ind_pml] - dEydz + dEydz / kappa);
        kpar->Hy[ind_ijk] = kpar->Hy[ind_ijk] - kpar->dt * (kpar->pml_Exz1[ind_pml] - dExdz + dExdz / kappa);
    }

    // update sources
    // kernel/domain's ind_global = i*_J*_K + j*_K + k;
    srcind_size_offset = 0;
    for(int ii = 0; ii < kpar->srclen; ii ++){
        srcind_size = kpar->Is[ii]*kpar->Js[ii]*kpar->Ks[ii];
        if( i >= kpar->i0s[ii] && j >= kpar->j0s[ii] && k >= kpar->k0s[ii]
            && i < kpar->i0s[ii]+kpar->Is[ii] && j < kpar->j0s[ii]+kpar->Js[ii] && k < kpar->k0s[ii]+kpar->Ks[ii]){
            srcind_src = (i-kpar->i0s[ii])*kpar->Js[ii]*kpar->Ks[ii] + (j-kpar->j0s[ii])*kpar->Ks[ii] + k-kpar->k0s[ii];
            if (kpar->t <= kpar->src_T){
                src_t = sin(kpar->t + kpar->Mx[srcind_src+srcind_size_offset].imag)*((1+kpar->src_min)*exp(-(kpar->t-kpar->src_T)*(kpar->t-kpar->src_T)/kpar->src_k)-kpar->src_min);
                kpar->Hx[ind_ijk] = kpar->Hx[ind_ijk] + src_t*kpar->Mx[srcind_src+srcind_size_offset].real*kpar->dt;
                src_t = sin(kpar->t + kpar->My[srcind_src+srcind_size_offset].imag)*((1+kpar->src_min)*exp(-(kpar->t-kpar->src_T)*(kpar->t-kpar->src_T)/kpar->src_k)-kpar->src_min);
                kpar->Hy[ind_ijk] = kpar->Hy[ind_ijk] + src_t*kpar->My[srcind_src+srcind_size_offset].real*kpar->dt;
                src_t = sin(kpar->t + kpar->Mz[srcind_src+srcind_size_offset].imag)*((1+kpar->src_min)*exp(-(kpar->t-kpar->src_T)*(kpar->t-kpar->src_T)/kpar->src_k)-kpar->src_min);
                kpar->Hz[ind_ijk] = kpar->Hz[ind_ijk] + src_t*kpar->Mz[srcind_src+srcind_size_offset].real*kpar->dt;
            }
            else{
                src_t = sin(kpar->t + kpar->Mx[srcind_src+srcind_size_offset].imag);
                kpar->Hx[ind_ijk] = kpar->Hx[ind_ijk] + src_t*kpar->Mx[srcind_src+srcind_size_offset].real*kpar->dt;
                src_t = sin(kpar->t + kpar->My[srcind_src+srcind_size_offset].imag);
                kpar->Hy[ind_ijk] = kpar->Hy[ind_ijk] + src_t*kpar->My[srcind_src+srcind_size_offset].real*kpar->dt;
                src_t = sin(kpar->t + kpar->Mz[srcind_src+srcind_size_offset].imag);
                kpar->Hz[ind_ijk] = kpar->Hz[ind_ijk] + src_t*kpar->Mz[srcind_src+srcind_size_offset].real*kpar->dt;
            }
        }
        srcind_size_offset += srcind_size;
    }
}

__global__ void kernel_update_E(kernelpar *kpar)
{
//     1D grid of 1D blocks
    size_t ind_global = blockIdx.x * blockDim.x + threadIdx.x;
//     1D grid of 2D blocks
//     size_t ind_global = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
//     2D grid of 1D blocks
//     size_t ind_global = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;

    if (ind_global >= kpar->size){
        return;
    }
    int i, j, k, ind_ijk, ind_ijm1k, ind_im1jk, ind_ijkm1;
    double dHxdy, dHxdz, dHydx, dHydz, dHzdx, dHzdy,
           b_x, b_y, b_z;
    int ind_pml, ind_pml_param;
    double b, C, kappa;
    int srcind_src,
        //srcind_ijk, srcind_global,
        srcind_size, srcind_size_offset;
    double src_t;

    // ind_global = i*_J*_K + j*_K + k;
    i = ind_global/(kpar->J*kpar->K);
    j = (ind_global%(kpar->J*kpar->K))/kpar->K;
    k = (ind_global%(kpar->J*kpar->K))%kpar->K;

    //blockDim.x = 128; gridDim.x=585
    // K=585, J=572, I=87
//     k = ind_global/(kpar->I*kpar->J);
//     j = (ind_global%(kpar->I*kpar->J))/kpar->I;
//     i = (ind_global%(kpar->I*kpar->J))%kpar->I;

    ind_ijk = (i+1)*(kpar->J+2)*(kpar->K+2) + (j+1)*(kpar->K+2) + k + 1;
    ind_ijm1k = ind_ijk - kpar->K - 2;
    ind_im1jk = ind_ijk - (kpar->J+2)*(kpar->K+2);
    ind_ijkm1 = ind_ijk - 1;

    // set up fields on the boundary
    if(kpar->k0==0 && k==0){
        if(kpar->bc[0]=='0'){
            kpar->Hy[ind_ijk-1] = 0.0;
            kpar->Hz[ind_ijk-1] = 0.0;
        }
        else if(kpar->bc[0]=='E'){
            kpar->Hy[ind_ijk-1] = -1.0*kpar->Hy[ind_ijk];
            kpar->Hz[ind_ijk-1] = -1.0*kpar->Hz[ind_ijk];
        }
        else if(kpar->bc[0]=='H'){
            kpar->Hy[ind_ijk-1] = kpar->Hy[ind_ijk];
            kpar->Hz[ind_ijk-1] = kpar->Hz[ind_ijk];
        }
    }

    if(kpar->j0==0 && j==0){
        if(kpar->bc[1]=='0'){
            kpar->Hx[ind_ijk-kpar->K-2] = 0.0;
            kpar->Hz[ind_ijk-kpar->K-2] = 0.0;
        }
        else if(kpar->bc[1]=='E'){
            kpar->Hx[ind_ijk-kpar->K-2] = -1.0*kpar->Hx[ind_ijk];
            kpar->Hz[ind_ijk-kpar->K-2] = -1.0*kpar->Hz[ind_ijk];
        }
        else if(kpar->bc[1]=='H'){
            kpar->Hx[ind_ijk-kpar->K-2] = kpar->Hx[ind_ijk];
            kpar->Hz[ind_ijk-kpar->K-2] = kpar->Hz[ind_ijk];
        }
    }

    if(kpar->i0==0 && i==0){
        if(kpar->bc[2]=='0'){
            kpar->Hx[ind_ijk-(kpar->J+2)*(kpar->K+2)] = 0.0;
            kpar->Hy[ind_ijk-(kpar->J+2)*(kpar->K+2)] = 0.0;
        }
        else if(kpar->bc[2]=='E'){
            kpar->Hx[ind_ijk-(kpar->J+2)*(kpar->K+2)] = -1.0*kpar->Hx[ind_ijk];
            kpar->Hy[ind_ijk-(kpar->J+2)*(kpar->K+2)] = -1.0*kpar->Hy[ind_ijk];
        }
        else if(kpar->bc[2]=='H'){
            kpar->Hx[ind_ijk-(kpar->J+2)*(kpar->K+2)] = kpar->Hx[ind_ijk];
            kpar->Hy[ind_ijk-(kpar->J+2)*(kpar->K+2)] = kpar->Hy[ind_ijk];
        }
    }

    // update fields
    b_x = kpar->dt/kpar->epsx[ind_global].real;
    b_y = kpar->dt/kpar->epsy[ind_global].real;
    b_z = kpar->dt/kpar->epsz[ind_global].real;
    dHzdy = kpar->odx * (kpar->Hz[ind_ijk] - kpar->Hz[ind_ijm1k]);
    dHydz = kpar->odx * (kpar->Hy[ind_ijk] - kpar->Hy[ind_im1jk]);
    dHxdz = kpar->odx * (kpar->Hx[ind_ijk] - kpar->Hx[ind_im1jk]);
    dHzdx = kpar->odx * (kpar->Hz[ind_ijk] - kpar->Hz[ind_ijkm1]);
    dHydx = kpar->odx * (kpar->Hy[ind_ijk] - kpar->Hy[ind_ijkm1]);
    dHxdy = kpar->odx * (kpar->Hx[ind_ijk] - kpar->Hx[ind_ijm1k]);
    kpar->Ex[ind_ijk] = kpar->Ex[ind_ijk] + b_x * (dHzdy - dHydz);
    kpar->Ey[ind_ijk] = kpar->Ey[ind_ijk] + b_y * (dHxdz - dHzdx);
    kpar->Ez[ind_ijk] = kpar->Ez[ind_ijk] + b_z * (dHydx - dHxdy);

    // Update PML
    if (k + kpar->k0 < kpar->pml_xmin){
        ind_pml = i * kpar->J * (kpar->pml_xmin - kpar->k0) + j * (kpar->pml_xmin - kpar->k0) + k;
        ind_pml_param = kpar->pml_xmin - k - kpar->k0 - 1;
        kappa = kpar->kappa_E_x[ind_pml_param];
        b = kpar->bEx[ind_pml_param];
        C = kpar->cEx[ind_pml_param];
        kpar->pml_Hyx0[ind_pml] = C * dHydx + b * kpar->pml_Hyx0[ind_pml];
        kpar->pml_Hzx0[ind_pml] = C * dHzdx + b * kpar->pml_Hzx0[ind_pml];
        kpar->Ez[ind_ijk] = kpar->Ez[ind_ijk] + b_z * (kpar->pml_Hyx0[ind_pml] - dHydx + dHydx / kappa);
        kpar->Ey[ind_ijk] = kpar->Ey[ind_ijk] - b_y * (kpar->pml_Hzx0[ind_pml] - dHzdx + dHzdx / kappa);
    }
    else if(k + kpar->k0 >= kpar->pml_xmax){
        ind_pml = i * kpar->J * (kpar->k0 + kpar->K - kpar->pml_xmax) + j * (kpar->k0 + kpar->K - kpar->pml_xmax)
                + k + kpar->k0 - kpar->pml_xmax;
        ind_pml_param = k + kpar->k0 - kpar->pml_xmax + kpar->w_pml_x0;
        kappa = kpar->kappa_E_x[ind_pml_param];
        b = kpar->bEx[ind_pml_param];
        C = kpar->cEx[ind_pml_param];
        kpar->pml_Hyx1[ind_pml] = C * dHydx + b * kpar->pml_Hyx1[ind_pml];
        kpar->pml_Hzx1[ind_pml] = C * dHzdx + b * kpar->pml_Hzx1[ind_pml];
        kpar->Ez[ind_ijk] = kpar->Ez[ind_ijk] + b_z * (kpar->pml_Hyx1[ind_pml] - dHydx + dHydx / kappa);
        kpar->Ey[ind_ijk] = kpar->Ey[ind_ijk] - b_y * (kpar->pml_Hzx1[ind_pml] - dHzdx + dHzdx / kappa);
    }

    if (j + kpar->j0 < kpar->pml_ymin){
        ind_pml = i * (kpar->pml_ymin - kpar->j0) * kpar->K + j * kpar->K + k;
        ind_pml_param = kpar->pml_ymin - j - kpar->j0 - 1;
        kappa = kpar->kappa_E_y[ind_pml_param];
        b = kpar->bEy[ind_pml_param];
        C = kpar->cEy[ind_pml_param];
        kpar->pml_Hxy0[ind_pml] = C * dHxdy + b * kpar->pml_Hxy0[ind_pml];
        kpar->pml_Hzy0[ind_pml] = C * dHzdy + b * kpar->pml_Hzy0[ind_pml];
        kpar->Ez[ind_ijk] = kpar->Ez[ind_ijk] - b_z * (kpar->pml_Hxy0[ind_pml] - dHxdy + dHxdy / kappa);
        kpar->Ex[ind_ijk] = kpar->Ex[ind_ijk] + b_x * (kpar->pml_Hzy0[ind_pml] - dHzdy + dHzdy / kappa);
    }
    else if(j + kpar->j0 >= kpar->pml_ymax){
        ind_pml = i * (kpar->j0 + kpar->J - kpar->pml_ymax) * kpar->K + (kpar->j0 + j - kpar->pml_ymax) * kpar->K + k;
        ind_pml_param = j + kpar->j0 - kpar->pml_ymax + kpar->w_pml_y0;
        kappa = kpar->kappa_E_y[ind_pml_param];
        b = kpar->bEy[ind_pml_param];
        C = kpar->cEy[ind_pml_param];
        kpar->pml_Hxy1[ind_pml] = C * dHxdy + b * kpar->pml_Hxy1[ind_pml];
        kpar->pml_Hzy1[ind_pml] = C * dHzdy + b * kpar->pml_Hzy1[ind_pml];
        kpar->Ez[ind_ijk] = kpar->Ez[ind_ijk] - b_z * (kpar->pml_Hxy1[ind_pml] - dHxdy + dHxdy / kappa);
        kpar->Ex[ind_ijk] = kpar->Ex[ind_ijk] + b_x * (kpar->pml_Hzy1[ind_pml] - dHzdy + dHzdy / kappa);
    }

    if (i + kpar->i0 < kpar->pml_zmin){
        ind_pml = i * kpar->J * kpar->K + j * kpar->K + k;
        ind_pml_param = kpar->pml_zmin - i - kpar->i0 - 1;
        kappa = kpar->kappa_E_z[ind_pml_param];
        b = kpar->bEz[ind_pml_param];
        C = kpar->cEz[ind_pml_param];
        kpar->pml_Hxz0[ind_pml] = C * dHxdz + b * kpar->pml_Hxz0[ind_pml];
        kpar->pml_Hyz0[ind_pml] = C * dHydz + b * kpar->pml_Hyz0[ind_pml];
        kpar->Ex[ind_ijk] = kpar->Ex[ind_ijk] - b_x * (kpar->pml_Hyz0[ind_pml] - dHydz + dHydz / kappa);
        kpar->Ey[ind_ijk] = kpar->Ey[ind_ijk] + b_y * (kpar->pml_Hxz0[ind_pml] - dHxdz + dHxdz / kappa);
    }
    else if(i + kpar->i0 > kpar->pml_zmax){
        ind_pml = (kpar->i0 + i - kpar->pml_zmax) * kpar->J * kpar->K + j * kpar->K + k;
        ind_pml_param = i + kpar->i0 - kpar->pml_zmax + kpar->w_pml_z0;
        kappa = kpar->kappa_E_z[ind_pml_param];
        b = kpar->bEz[ind_pml_param];
        C = kpar->cEz[ind_pml_param];
        kpar->pml_Hxz1[ind_pml] = C * dHxdz + b * kpar->pml_Hxz1[ind_pml];
        kpar->pml_Hyz1[ind_pml] = C * dHydz + b * kpar->pml_Hyz1[ind_pml];
        kpar->Ex[ind_ijk] = kpar->Ex[ind_ijk] - b_x * (kpar->pml_Hyz1[ind_pml] - dHydz + dHydz / kappa);
        kpar->Ey[ind_ijk] = kpar->Ey[ind_ijk] + b_y * (kpar->pml_Hxz1[ind_pml] - dHxdz + dHxdz / kappa);
    }

    // update sources
    // kernel/domain's ind_global = i*_J*_K + j*_K + k;
    srcind_size_offset = 0;
    for(int ii = 0; ii < kpar->srclen; ii++){
        srcind_size = kpar->Is[ii] * kpar->Js[ii] * kpar->Ks[ii];
        if(i >= kpar->i0s[ii] && j >= kpar->j0s[ii] && k >= kpar->k0s[ii]
           && i < kpar->i0s[ii]+kpar->Is[ii] && j < kpar->j0s[ii]+kpar->Js[ii] && k < kpar->k0s[ii]+kpar->Ks[ii]){
            srcind_src = (i-kpar->i0s[ii])*kpar->Js[ii]*kpar->Ks[ii] + (j-kpar->j0s[ii])*kpar->Ks[ii] + k - kpar->k0s[ii];
            if (kpar->t <= kpar->src_T){
                src_t = sin(kpar->t + kpar->Jx[srcind_src+srcind_size_offset].imag)*((1+kpar->src_min)*exp(-(kpar->t-kpar->src_T)*(kpar->t-kpar->src_T)/kpar->src_k)-kpar->src_min);
                kpar->Ex[ind_ijk] = kpar->Ex[ind_ijk] - src_t*kpar->Jx[srcind_src+srcind_size_offset].real*kpar->dt/kpar->epsx[ind_global].real;
                src_t = sin(kpar->t + kpar->Jy[srcind_src+srcind_size_offset].imag)*((1+kpar->src_min)*exp(-(kpar->t-kpar->src_T)*(kpar->t-kpar->src_T)/kpar->src_k)-kpar->src_min);
                kpar->Ey[ind_ijk] = kpar->Ey[ind_ijk] - src_t*kpar->Jy[srcind_src+srcind_size_offset].real*kpar->dt/kpar->epsy[ind_global].real;
                src_t = sin(kpar->t + kpar->Jz[srcind_src+srcind_size_offset].imag)*((1+kpar->src_min)*exp(-(kpar->t-kpar->src_T)*(kpar->t-kpar->src_T)/kpar->src_k)-kpar->src_min);
                kpar->Ez[ind_ijk] = kpar->Ez[ind_ijk] - src_t*kpar->Jz[srcind_src+srcind_size_offset].real*kpar->dt/kpar->epsz[ind_global].real;
            }
            else{
                src_t = sin(kpar->t + kpar->Jx[srcind_src+srcind_size_offset].imag);
                kpar->Ex[ind_ijk] = kpar->Ex[ind_ijk] - src_t*kpar->Jx[srcind_src+srcind_size_offset].real*kpar->dt/kpar->epsx[ind_global].real;
                src_t = sin(kpar->t + kpar->Jy[srcind_src+srcind_size_offset].imag);
                kpar->Ey[ind_ijk] = kpar->Ey[ind_ijk] - src_t*kpar->Jy[srcind_src+srcind_size_offset].real*kpar->dt/kpar->epsy[ind_global].real;
                src_t = sin(kpar->t + kpar->Jz[srcind_src+srcind_size_offset].imag);
                kpar->Ez[ind_ijk] = kpar->Ez[ind_ijk] - src_t*kpar->Jz[srcind_src+srcind_size_offset].real*kpar->dt/kpar->epsz[ind_global].real;
           }
        }
        srcind_size_offset += srcind_size;
    }
}

__global__ void kernel_calc_ydAx(size_t size, size_t Nx, size_t Ny, size_t Nz, size_t i0, size_t i1,
                thrust::complex<double> *ydAx,
                thrust::complex<double> *Ex_adj, thrust::complex<double> *Ey_adj, thrust::complex<double> *Ez_adj,
                thrust::complex<double> *Ex_fwd, thrust::complex<double> *Ey_fwd, thrust::complex<double> *Ez_fwd,
                thrust::complex<double> *epsx0, thrust::complex<double> *epsy0, thrust::complex<double> *epsz0,
                thrust::complex<double> *epsxp, thrust::complex<double> *epsyp, thrust::complex<double> *epszp)
{
    int i,j,k;

    k = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    i = blockIdx.z;
    if( k >=Nx || j>=Ny || i>=Nz ) { return; }
    size_t ind = (i1-i0)*Nx*Ny + j *Nx+k;
    size_t offset = Nx * Ny * i0;
    if(ind >= size) { return; }

    ydAx[ind] = Ex_adj[ind] * Ex_fwd[ind] * (epsxp[ind+offset]-epsx0[ind+offset]) +
                Ey_adj[ind] * Ey_fwd[ind] * (epsyp[ind+offset]-epsy0[ind+offset]) +
                Ez_adj[ind] * Ez_fwd[ind] * (epszp[ind+offset]-epsz0[ind+offset]);
}

void fdtd::FDTD::calc_ydAx(size_t size, size_t Nx, size_t Ny, size_t Nz, size_t i0, size_t i1, size_t i2,
                std::complex<double> *ydAx,
                std::complex<double> *Ex_adj, std::complex<double> *Ey_adj, std::complex<double> *Ez_adj,
                std::complex<double> *Ex_fwd, std::complex<double> *Ey_fwd, std::complex<double> *Ez_fwd,
                std::complex<double> *epsx0, std::complex<double> *epsy0, std::complex<double> *epsz0,
                std::complex<double> *epsxp, std::complex<double> *epsyp, std::complex<double> *epszp)
{
    thrust::complex<double> *dydAx, *dEx_adj, *dEy_adj, *dEz_adj, *dEx_fwd, *dEy_fwd, *dEz_fwd,
                            *depsx0, *depsy0, *depsz0, *depsxp, *depsyp, *depszp;
    size_t sizefull = Nx * Ny * Nz;
    gpuErrchk(cudaMalloc((void **)&dydAx, size * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&dEx_adj, size * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&dEy_adj, size * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&dEz_adj, size * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&dEx_fwd, size * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&dEy_fwd, size * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&dEz_fwd, size * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&depsx0, sizefull * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&depsy0, sizefull * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&depsz0, sizefull * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&depsxp, sizefull * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&depsyp, sizefull * sizeof(std::complex<double>)));
    gpuErrchk(cudaMalloc((void **)&depszp, sizefull * sizeof(std::complex<double>)));

    gpuErrchk(cudaMemcpy(dEx_adj, Ex_adj, size * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEy_adj, Ey_adj, size * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEz_adj, Ez_adj, size * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEx_fwd, Ex_fwd, size * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEy_fwd, Ey_fwd, size * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dEz_fwd, Ez_fwd, size * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(depsx0, epsx0, sizefull * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(depsy0, epsy0, sizefull * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(depsz0, epsz0, sizefull * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(depsxp, epsxp, sizefull * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(depsyp, epsyp, sizefull * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(depszp, epszp, sizefull * sizeof(std::complex<double>), cudaMemcpyHostToDevice));

//    kernel_calc_ydAx <<< ceil(size/128.0), 128 >>> (size, NxNy, i0, i1, dydAx, dEx_adj, dEy_adj, dEz_adj, dEx_fwd, dEy_fwd, dEz_fwd,
//                depsx0, depsy0, depsz0, depsxp, depsyp, depszp);

    dim3 block_dim(128,1);
    dim3 grid_dim((int)ceil(Nx/128.0), (int)ceil(Ny/1.0));
    for(int i=i1; i<i2; i++){
        kernel_calc_ydAx <<< grid_dim, block_dim >>> (size, Nx, Ny, Nz, i1, i, dydAx, dEx_adj, dEy_adj, dEz_adj, dEx_fwd, dEy_fwd, dEz_fwd,
                depsx0, depsy0, depsz0, depsxp, depsyp, depszp);
    }

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if(cudaerr != cudaSuccess){ printf("fdtd::calc_ydAx: kernel launch status \"%s\".\n", cudaGetErrorString(cudaerr)); }

    gpuErrchk(cudaMemcpy(ydAx, dydAx, size * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(dydAx));
    gpuErrchk(cudaFree(dEx_adj)); gpuErrchk(cudaFree(dEy_adj)); gpuErrchk(cudaFree(dEz_adj));
    gpuErrchk(cudaFree(dEx_fwd)); gpuErrchk(cudaFree(dEy_fwd)); gpuErrchk(cudaFree(dEz_fwd));
    gpuErrchk(cudaFree(depsx0)); gpuErrchk(cudaFree(depsy0)); gpuErrchk(cudaFree(depsz0));
    gpuErrchk(cudaFree(depsxp)); gpuErrchk(cudaFree(depsyp)); gpuErrchk(cudaFree(depszp));
}

fdtd::FDTD::FDTD()
{
    // make sure all of our PML arrays start NULL
    _pml_Exy0 = NULL; _pml_Exy1 = NULL; _pml_Exz0 = NULL; _pml_Exz1 = NULL;
    _pml_Eyx0 = NULL; _pml_Eyx1 = NULL; _pml_Eyz0 = NULL; _pml_Eyz1 = NULL;
    _pml_Ezx0 = NULL; _pml_Ezx1 = NULL; _pml_Ezy0 = NULL; _pml_Ezy1 = NULL;
    _pml_Hxy0 = NULL; _pml_Hxy1 = NULL; _pml_Hxz0 = NULL; _pml_Hxz1 = NULL;
    _pml_Hyx0 = NULL; _pml_Hyx1 = NULL; _pml_Hyz0 = NULL; _pml_Hyz1 = NULL;
    _pml_Hzx0 = NULL; _pml_Hzx1 = NULL; _pml_Hzy0 = NULL; _pml_Hzy1 = NULL;
    
    _kappa_H_x = NULL; _kappa_H_y = NULL; _kappa_H_z = NULL;
    _kappa_E_x = NULL; _kappa_E_y = NULL; _kappa_E_z = NULL;

    _bHx = NULL; _bHy = NULL; _bHz = NULL;
    _bEx = NULL; _bEy = NULL; _bEz = NULL;

    _cHx = NULL; _cHy = NULL; _cHz = NULL;
    _cEx = NULL; _cEy = NULL; _cEz = NULL;

    _w_pml_x0 = 0; _w_pml_x1 = 0;
    _w_pml_y0 = 0; _w_pml_y1 = 0;
    _w_pml_z0 = 0; _w_pml_z1 = 0;

    _complex_eps = false;

    _kpar_host = (kernelpar *)malloc(sizeof(kernelpar));
    _kpar_host->srclen = 0;
    _srclen = 0;
    // kernel parameter structures
    gpuErrchk(cudaMalloc((void **)&_kpar_device, sizeof(kernelpar)));

}

fdtd::FDTD::~FDTD()
{
    // Clean up PML arrays
    delete[] _pml_Exy0; delete[] _pml_Exy1; delete[] _pml_Exz0; delete[] _pml_Exz1;
    delete[] _pml_Eyx0; delete[] _pml_Eyx1; delete[] _pml_Eyz0; delete[] _pml_Eyz1;
    delete[] _pml_Ezx0; delete[] _pml_Ezx1; delete[] _pml_Ezy0; delete[] _pml_Ezy1;
    delete[] _pml_Hxy0; delete[] _pml_Hxy1; delete[] _pml_Hxz0; delete[] _pml_Hxz1;
    delete[] _pml_Hyx0; delete[] _pml_Hyx1; delete[] _pml_Hyz0; delete[] _pml_Hyz1;
    delete[] _pml_Hzx0; delete[] _pml_Hzx1; delete[] _pml_Hzy0; delete[] _pml_Hzy1;

    delete [] _kappa_H_x;
    delete [] _kappa_H_y;
    delete [] _kappa_H_z;

    delete [] _kappa_E_x;
    delete [] _kappa_E_y;
    delete [] _kappa_E_z;

    delete [] _bHx;
    delete [] _bHy;
    delete [] _bHz;

    delete [] _bEx;
    delete [] _bEy;
    delete [] _bEz;

    delete [] _cHx;
    delete [] _cHy;
    delete [] _cHz;

    delete [] _cEx;
    delete [] _cEy;
    delete [] _cEz;

    delete [] _kpar_host;

    cudaFree(_kpar_device);

}

void fdtd::FDTD::set_physical_dims(double X, double Y, double Z,
                                         double dx, double dy, double dz)
{
    _X = X; _Y = Y; _Z = Z;
    _dx = dx; _dy = dy; _dz = dz;
}

void fdtd::FDTD::set_grid_dims(int Nx, int Ny, int Nz)
{
    _Nx = Nx;
    _Ny = Ny;
    _Nz = Nz;
}


void fdtd::FDTD::set_local_grid(int k0, int j0, int i0, int K, int J, int I)
{
    _i0 = i0; _j0 = j0; _k0 = k0;
    _I = I; _J = J; _K = K;

}

void fdtd::FDTD::set_local_grid_perturb(int i1, int i2)
{
    _i1 = i1; _i2 = i2;
}
void fdtd::FDTD::set_wavelength(double wavelength)
{
    _wavelength = wavelength;
    _R = _wavelength/(2*M_PI);
}


void fdtd::FDTD::set_dt(double dt)
{
    _dt = dt;
    _odt = 1.0/_dt;
}

void fdtd::FDTD::set_complex_eps(bool complex_eps)
{
    _complex_eps = complex_eps;
}

void fdtd::FDTD::copyCUDA_field_arrays()
{
    size_t size = (_kpar_host->I+2)*(_kpar_host->J+2)*(_kpar_host->K+2);
    gpuErrchk(cudaMemcpy(_Ex, _kpar_host->Ex, size * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(_Ey, _kpar_host->Ey, size * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(_Ez, _kpar_host->Ez, size * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(_Hx, _kpar_host->Hx, size * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(_Hy, _kpar_host->Hy, size * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(_Hz, _kpar_host->Hz, size * sizeof(double), cudaMemcpyDeviceToHost));
}

void fdtd::FDTD::block_CUDA_free()
{
    // fields
    cudaFree(dEx);cudaFree(dEy);cudaFree(dEz);cudaFree(dHx);cudaFree(dHy);cudaFree(dHz);
    // materials
    cudaFree(depsx);cudaFree(depsy);cudaFree(depsz);
    // PML
    cudaFree(dpml_Exy0);cudaFree(dpml_Exy1);cudaFree(dpml_Exz0);cudaFree(dpml_Exz1);
    cudaFree(dpml_Eyx0);cudaFree(dpml_Eyx1);cudaFree(dpml_Eyz0);cudaFree(dpml_Eyz1);
    cudaFree(dpml_Ezx0);cudaFree(dpml_Ezx1);cudaFree(dpml_Ezy0);cudaFree(dpml_Ezy1);
    cudaFree(dpml_Hxy0);cudaFree(dpml_Hxy1);cudaFree(dpml_Hxz0);cudaFree(dpml_Hxz1);
    cudaFree(dpml_Hyx0);cudaFree(dpml_Hyx1);cudaFree(dpml_Hyz0);cudaFree(dpml_Hyz1);
    cudaFree(dpml_Hzx0);cudaFree(dpml_Hzx1);cudaFree(dpml_Hzy0);cudaFree(dpml_Hzy1);

    cudaFree(dkappa_H_x);cudaFree(dkappa_H_y);cudaFree(dkappa_H_z);
    cudaFree(dkappa_E_x);cudaFree(dkappa_E_y);cudaFree(dkappa_E_z);
    cudaFree(dbHx);cudaFree(dbHy);cudaFree(dbHz);
    cudaFree(dbEx);cudaFree(dbEy);cudaFree(dbEz);
    cudaFree(dcHx);cudaFree(dcHy);cudaFree(dcHz);
    cudaFree(dcEx);cudaFree(dcEy);cudaFree(dcEz);

}

void fdtd::FDTD::block_CUDA_src_free()
{
    // source index arrays
    gpuErrchk(cudaFree(di0s));
    cudaFree(dj0s);
    cudaFree(dk0s);
    cudaFree(dIs);
    cudaFree(dJs);
    cudaFree(dKs);

    // source arrays
    gpuErrchk(cudaFree(dJx));
    cudaFree(dJy);
    cudaFree(dJz);
    cudaFree(dMx);
    cudaFree(dMy);
    cudaFree(dMz);
}
void fdtd::FDTD::block_CUDA_src_malloc_memcpy()
{
    // extract the list of tuples of the sources
    _i0s = (int *)malloc(_srclen * sizeof(int));
    _j0s = (int *)malloc(_srclen * sizeof(int));
    _k0s = (int *)malloc(_srclen * sizeof(int));
    _Is = (int *)malloc(_srclen * sizeof(int));
    _Js = (int *)malloc(_srclen * sizeof(int));
    _Ks = (int *)malloc(_srclen * sizeof(int));
    size_t size = 0,
           size_offset = 0,
           sizeall = 0;
    for(int i=0; i < _srclen; i++){
        _i0s[i] = _sources[i].i0;
        _j0s[i] = _sources[i].j0;
        _k0s[i] = _sources[i].k0;
        _Is[i] = _sources[i].I;
        _Js[i] = _sources[i].J;
        _Ks[i] = _sources[i].K;
        sizeall += _Is[i] * _Js[i] * _Ks[i];
    }

    // initialize GPU memory to store source arrays
    gpuErrchk(cudaMalloc((void **)&dJx, sizeall * sizeof(complex128)));
    cudaMalloc((void **)&dJy, sizeall * sizeof(complex128));
    cudaMalloc((void **)&dJz, sizeall * sizeof(complex128));
    cudaMalloc((void **)&dMx, sizeall * sizeof(complex128));
    cudaMalloc((void **)&dMy, sizeall * sizeof(complex128));
    cudaMalloc((void **)&dMz, sizeall * sizeof(complex128));

    gpuErrchk(cudaMalloc((void **)&di0s, _srclen * sizeof(int)));
    cudaMalloc((void **)&dj0s, _srclen * sizeof(int));
    cudaMalloc((void **)&dk0s, _srclen * sizeof(int));
    cudaMalloc((void **)&dIs, _srclen * sizeof(int));
    cudaMalloc((void **)&dJs, _srclen * sizeof(int));
    cudaMalloc((void **)&dKs, _srclen * sizeof(int));

    gpuErrchk(cudaMemcpy(di0s, _i0s, _srclen * sizeof(int), cudaMemcpyHostToDevice));
    cudaMemcpy(dj0s, _j0s, _srclen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dk0s, _k0s, _srclen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dIs, _Is, _srclen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJs, _Js, _srclen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dKs, _Ks, _srclen * sizeof(int), cudaMemcpyHostToDevice);

    size_offset = 0;
    for(int i=0; i < _srclen; i++){
        size = _Is[i] * _Js[i] * _Ks[i];
        gpuErrchk(cudaMemcpy(dJx+size_offset, _sources[i].Jx, size * sizeof(complex128), cudaMemcpyHostToDevice));
        cudaMemcpy(dJy+size_offset, _sources[i].Jy, size * sizeof(complex128), cudaMemcpyHostToDevice);
        cudaMemcpy(dJz+size_offset, _sources[i].Jz, size * sizeof(complex128), cudaMemcpyHostToDevice);
        cudaMemcpy(dMx+size_offset, _sources[i].Mx, size * sizeof(complex128), cudaMemcpyHostToDevice);
        cudaMemcpy(dMy+size_offset, _sources[i].My, size * sizeof(complex128), cudaMemcpyHostToDevice);
        cudaMemcpy(dMz+size_offset, _sources[i].Mz, size * sizeof(complex128), cudaMemcpyHostToDevice);
        size_offset += size;
    }

    _kpar_host->i0s = di0s;
    _kpar_host->j0s = dj0s;
    _kpar_host->k0s = dk0s;
    _kpar_host->Is = dIs;
    _kpar_host->Js = dJs;
    _kpar_host->Ks = dKs;
    _kpar_host->Jx = dJx;
    _kpar_host->Jy = dJy;
    _kpar_host->Jz = dJz;
    _kpar_host->Mx = dMx;
    _kpar_host->My = dMy;
    _kpar_host->Mz = dMz;

}

void fdtd::FDTD::block_CUDA_malloc_memcpy()
{
    // BCs
    gpuErrchk(cudaMalloc((void **)&dbc, 3 * sizeof(char)));
    gpuErrchk(cudaMemcpy(dbc, _bc, 3 * sizeof(char), cudaMemcpyHostToDevice));
    _kpar_host->bc = dbc;

    // fields
    size_t size = (_I+2)*(_J+2)*(_K+2);
    gpuErrchk(cudaMalloc((void **)&dEx, size * sizeof(double)));
    cudaMalloc((void **)&dEy, size * sizeof(double));
    cudaMalloc((void **)&dEz, size * sizeof(double));
    cudaMalloc((void **)&dHx, size * sizeof(double));
    cudaMalloc((void **)&dHy, size * sizeof(double));
    cudaMalloc((void **)&dHz, size * sizeof(double));

    gpuErrchk(cudaMemcpy(dEx, _Ex, size * sizeof(double), cudaMemcpyHostToDevice));
    cudaMemcpy(dEy, _Ey, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dEz, _Ez, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dHx, _Hx, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dHy, _Hy, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dHz, _Hz, size * sizeof(double), cudaMemcpyHostToDevice);

    _kpar_host->Ex = dEx;
    _kpar_host->Ey = dEy;
    _kpar_host->Ez = dEz;
    _kpar_host->Hx = dHx;
    _kpar_host->Hy = dHy;
    _kpar_host->Hz = dHz;

    _kpar_host->I = _I;
    _kpar_host->J = _J;
    _kpar_host->K = _K;
    _kpar_host->i0 = _i0;
    _kpar_host->j0 = _j0;
    _kpar_host->k0 = _k0;
    _kpar_host->Nx = _Nx;
    _kpar_host->Ny = _Ny;
    _kpar_host->Nz = _Nz;
    _kpar_host->size = _I*_J*_K;

    _kpar_host->dt = _dt;

    // materials
    size = _I * _J * _K;
    gpuErrchk(cudaMalloc((void **)&depsx, size * sizeof(complex128)));
    cudaMalloc((void **)&depsy, size * sizeof(complex128));
    cudaMalloc((void **)&depsz, size * sizeof(complex128));

    gpuErrchk(cudaMemcpy(depsx, _eps_x, size * sizeof(complex128), cudaMemcpyHostToDevice));
    cudaMemcpy(depsy, _eps_y, size * sizeof(complex128), cudaMemcpyHostToDevice);
    cudaMemcpy(depsz, _eps_z, size * sizeof(complex128), cudaMemcpyHostToDevice);

    _kpar_host->epsx = depsx;
    _kpar_host->epsy = depsy;
    _kpar_host->epsz = depsz;

    // PML
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,
        zmin = _w_pml_z0, zmax = _Nz-_w_pml_z1;
    int Npmlx = _w_pml_x0 + _w_pml_x1,
        Npmly = _w_pml_y0 + _w_pml_y1,
        Npmlz = _w_pml_z0 + _w_pml_z1;

    // touches xmin boudary
    if(_k0 < xmin) {
        N = _I * _J * (xmin - _k0);
        gpuErrchk(cudaMalloc((void **)&dpml_Eyx0, N * sizeof(double)));
        cudaMalloc((void **)&dpml_Ezx0, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hyx0, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hzx0, N * sizeof(double));
        gpuErrchk(cudaMemset(dpml_Eyx0, 0.0, N * sizeof(double)));
        cudaMemset(dpml_Ezx0, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hyx0, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hzx0, 0.0, N * sizeof(double));
    }
    // touches xmax boundary
    if(_k0 +_K > xmax) {
        N = _I * _J * (_k0  + _K - xmax);
        gpuErrchk(cudaMalloc((void **)&dpml_Eyx1, N * sizeof(double)));
        cudaMalloc((void **)&dpml_Ezx1, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hyx1, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hzx1, N * sizeof(double));
        gpuErrchk(cudaMemset(dpml_Eyx1, 0.0, N * sizeof(double)));
        cudaMemset(dpml_Ezx1, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hyx1, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hzx1, 0.0, N * sizeof(double));
    }
    // touches ymin boundary
    if(_j0 < ymin) {
        N = _I * _K * (ymin - _j0);
        gpuErrchk(cudaMalloc((void **)&dpml_Exy0, N * sizeof(double)));
        cudaMalloc((void **)&dpml_Ezy0, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hxy0, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hzy0, N * sizeof(double));
        gpuErrchk(cudaMemset(dpml_Exy0, 0.0, N * sizeof(double)));
        cudaMemset(dpml_Ezy0, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hxy0, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hzy0, 0.0, N * sizeof(double));
    }
    // touches ymax boundary
    if(_j0 + _J > ymax) {
        N = _I * _K * (_j0 + _J - ymax);
        gpuErrchk(cudaMalloc((void **)&dpml_Exy1, N * sizeof(double)));
        cudaMalloc((void **)&dpml_Ezy1, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hxy1, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hzy1, N * sizeof(double));
        gpuErrchk(cudaMemset(dpml_Exy1, 0.0, N * sizeof(double)));
        cudaMemset(dpml_Ezy1, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hxy1, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hzy1, 0.0, N * sizeof(double));
    }
    // touches zmin boundary
    if(_i0 < zmin) {
        N = _J * _K * (zmin - _i0);
        gpuErrchk(cudaMalloc((void **)&dpml_Exz0, N * sizeof(double)));
        cudaMalloc((void **)&dpml_Eyz0, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hxz0, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hyz0, N * sizeof(double));
        gpuErrchk(cudaMemset(dpml_Exz0, 0.0, N * sizeof(double)));
        cudaMemset(dpml_Eyz0, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hxz0, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hyz0, 0.0, N * sizeof(double));
    }
    // touches zmax boundary
    if(_i0 + _I > zmax) {
        N = _J * _K * (_i0 + _I - zmax);
        gpuErrchk(cudaMalloc((void **)&dpml_Exz1, N * sizeof(double)));
        cudaMalloc((void **)&dpml_Eyz1, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hxz1, N * sizeof(double));
        cudaMalloc((void **)&dpml_Hyz1, N * sizeof(double));
        gpuErrchk(cudaMemset(dpml_Exz1, 0.0, N * sizeof(double)));
        cudaMemset(dpml_Eyz1, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hxz1, 0.0, N * sizeof(double));
        cudaMemset(dpml_Hyz1, 0.0, N * sizeof(double));
    }

    gpuErrchk(cudaMalloc((void **)&dkappa_H_x, Npmlx * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dkappa_H_y, Npmly * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dkappa_H_z, Npmlz * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dkappa_E_x, Npmlx * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dkappa_E_y, Npmly * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dkappa_E_z, Npmlz * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dbHx, Npmlx * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dbHy, Npmly * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dbHz, Npmlz * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dbEx, Npmlx * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dbEy, Npmly * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dbEz, Npmlz * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dcHx, Npmlx * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dcHy, Npmly * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dcHz, Npmlz * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dcEx, Npmlx * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dcEy, Npmly * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&dcEz, Npmlz * sizeof(double)));

    gpuErrchk(cudaMemcpy(dkappa_H_x, _kappa_H_x, Npmlx * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dkappa_H_y, _kappa_H_y, Npmly * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dkappa_H_z, _kappa_H_z, Npmlz * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dkappa_E_x, _kappa_E_x, Npmlx * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dkappa_E_y, _kappa_E_y, Npmly * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dkappa_E_z, _kappa_E_z, Npmlz * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dbHx, _bHx, Npmlx * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dbHy, _bHy, Npmly * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dbHz, _bHz, Npmlz * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dbEx, _bEx, Npmlx * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dbEy, _bEy, Npmly * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dbEz, _bEz, Npmlz * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dcHx, _cHx, Npmlx * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dcHy, _cHy, Npmly * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dcHz, _cHz, Npmlz * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dcEx, _cEx, Npmlx * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dcEy, _cEy, Npmly * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dcEz, _cEz, Npmlz * sizeof(double), cudaMemcpyHostToDevice));

    _kpar_host->pml_Exy0 = dpml_Exy0;
    _kpar_host->pml_Exy1 = dpml_Exy1;
    _kpar_host->pml_Exz0 = dpml_Exz0;
    _kpar_host->pml_Exz1 = dpml_Exz1;
    _kpar_host->pml_Eyx0 = dpml_Eyx0;
    _kpar_host->pml_Eyx1 = dpml_Eyx1;
    _kpar_host->pml_Eyz0 = dpml_Eyz0;
    _kpar_host->pml_Eyz1 = dpml_Eyz1;
    _kpar_host->pml_Ezx0 = dpml_Ezx0;
    _kpar_host->pml_Ezx1 = dpml_Ezx1;
    _kpar_host->pml_Ezy0 = dpml_Ezy0;
    _kpar_host->pml_Ezy1 = dpml_Ezy1;

    _kpar_host->pml_Hxy0 = dpml_Hxy0;
    _kpar_host->pml_Hxy1 = dpml_Hxy1;
    _kpar_host->pml_Hxz0 = dpml_Hxz0;
    _kpar_host->pml_Hxz1 = dpml_Hxz1;
    _kpar_host->pml_Hyx0 = dpml_Hyx0;
    _kpar_host->pml_Hyx1 = dpml_Hyx1;
    _kpar_host->pml_Hyz0 = dpml_Hyz0;
    _kpar_host->pml_Hyz1 = dpml_Hyz1;
    _kpar_host->pml_Hzx0 = dpml_Hzx0;
    _kpar_host->pml_Hzx1 = dpml_Hzx1;
    _kpar_host->pml_Hzy0 = dpml_Hzy0;
    _kpar_host->pml_Hzy1 = dpml_Hzy1;

    _kpar_host->kappa_H_x = dkappa_H_x;
    _kpar_host->kappa_H_y = dkappa_H_y;
    _kpar_host->kappa_H_z = dkappa_H_z;
    _kpar_host->kappa_E_x = dkappa_E_x;
    _kpar_host->kappa_E_y = dkappa_E_y;
    _kpar_host->kappa_E_z = dkappa_E_z;
    _kpar_host->bHx = dbHx;
    _kpar_host->bHy = dbHy;
    _kpar_host->bHz = dbHz;
    _kpar_host->bEx = dbEx;
    _kpar_host->bEy = dbEy;
    _kpar_host->bEz = dbEz;
    _kpar_host->cHx = dcHx;
    _kpar_host->cHy = dcHy;
    _kpar_host->cHz = dcHz;
    _kpar_host->cEx = dcEx;
    _kpar_host->cEy = dcEy;
    _kpar_host->cEz = dcEz;

    _kpar_host->w_pml_x0 = _w_pml_x0;
    _kpar_host->w_pml_x1 = _w_pml_x1;
    _kpar_host->w_pml_y0 = _w_pml_y0;
    _kpar_host->w_pml_y1 = _w_pml_y1;
    _kpar_host->w_pml_z0 = _w_pml_z0;
    _kpar_host->w_pml_z1 = _w_pml_z1;

//Test cudaMalloc and cudaMemcpy
    //gpuErrchk(cudaMemcpy(_kpar_device,_kpar_host, sizeof(kernelpar), cudaMemcpyHostToDevice));
    //kernel_test_pointer <<< ceil(size/256.0), 256 >>> (_kpar_device);
    //cudaError_t cudaerr = cudaDeviceSynchronize();
    //if (cudaerr != cudaSuccess)
    //    printf("Set field: kernel launch status \"%s\".\n", cudaGetErrorString(cudaerr));
    //gpuErrchk(cudaMemcpy(_Ex, _kpar_host->Ex, size * sizeof(double), cudaMemcpyDeviceToHost));
    //printf("set field %d\n", _kpar_host->I);

//Test cudaMalloc and cudaMemcpy
    //kernel_test_double <<< ceil(size/256.0), 256 >>> (dEx, size);
    //cudaError_t cudaerr = cudaDeviceSynchronize();
    //if (cudaerr != cudaSuccess)
    //    printf("Set field: kernel launch status \"%s\".\n", cudaGetErrorString(cudaerr));
    //cudaMemcpy(Ex, dEx, size * sizeof(double), cudaMemcpyDeviceToHost);

// Test passing param struct to kernel_update_H
//     cudaMemcpy(_kpar_device,_kpar_host, sizeof(kernelpar), cudaMemcpyHostToDevice);
//     kernel_update_H <<< ceil(size/256.0), 256 >>> (_kpar_device);
//     cudaError_t cudaerr = cudaDeviceSynchronize();
//     printf("Set field: kernel launch status \"%s\".\n", cudaGetErrorString(cudaerr));
//     cudaMemcpy(Ex, dEx, size * sizeof(double), cudaMemcpyDeviceToHost);

}

void fdtd::FDTD::set_field_arrays(double *Ex, double *Ey, double *Ez,
                                  double *Hx, double *Hy, double *Hz)
{
    _Ex = Ex; _Ey = Ey; _Ez = Ez;
    _Hx = Hx; _Hy = Hy; _Hz = Hz;

}

void fdtd::FDTD::set_mat_arrays(complex128 *eps_x, complex128 *eps_y, complex128 *eps_z)
{
    _eps_x = eps_x; _eps_y = eps_y; _eps_z = eps_z;
//    _mu_x = mu_x; _mu_y = mu_y; _mu_z = mu_z;

}

void fdtd::FDTD::update_H(int n, double t)
{
    double odx = _R/_dx,
           ody = _R/_dy,
           odz = _R/_dz;

    int pml_xmin = _w_pml_x0, pml_xmax = _Nx-_w_pml_x1,
        pml_ymin = _w_pml_y0, pml_ymax = _Ny-_w_pml_y1,
        pml_zmin = _w_pml_z0, pml_zmax = _Nz-_w_pml_z1;

    _kpar_host->pml_xmin = pml_xmin;
    _kpar_host->pml_xmax = pml_xmax;
    _kpar_host->pml_ymin = pml_ymin;
    _kpar_host->pml_ymax = pml_ymax;
    _kpar_host->pml_zmin = pml_zmin;
    _kpar_host->pml_zmax = pml_zmax;
    _kpar_host->odx = odx;
    _kpar_host->t = t;
    _kpar_host->src_T = _src_T;
    _kpar_host->src_min = _src_min;
    _kpar_host->src_k = _src_k;

//     dim3 block(16,8);
//     dim3 grid(ceil(_I*_J*_K/128.0/572.0), 572);

//     printf("block dim: %d\n", blkdim);
//     printf("grid dim: %d\n", blkdim/585);
//     printf("grid length: %d\n", blkdim%585);

    gpuErrchk(cudaMemcpy(_kpar_device, _kpar_host, sizeof(kernelpar), cudaMemcpyHostToDevice));
    kernel_update_H <<< ceil(_I*_J*_K/128.0), 128 >>> (_kpar_device);
//     kernel_update_H <<< grid, 128 >>> (_kpar_device);
//     cudaError_t cudaerr = cudaDeviceSynchronize();
//     printf("update H: kernel launch status \"%s\".\n", cudaGetErrorString(cudaerr));

//     auto start3 = std::chrono::system_clock::now();
//     auto end3 = std::chrono::system_clock::now();
//     std::chrono::duration<double> elapsed_src = end3-start3;
//     std::cout << "Src:" << elapsed_src.count();
}

void fdtd::FDTD::update_E(int n, double t)
{
    double odx = _R/_dx;

    int pml_xmin = _w_pml_x0, pml_xmax = _Nx-_w_pml_x1,
        pml_ymin = _w_pml_y0, pml_ymax = _Ny-_w_pml_y1,
        pml_zmin = _w_pml_z0, pml_zmax = _Nz-_w_pml_z1;

    _kpar_host->pml_xmin = pml_xmin;
    _kpar_host->pml_xmax = pml_xmax;
    _kpar_host->pml_ymin = pml_ymin;
    _kpar_host->pml_ymax = pml_ymax;
    _kpar_host->pml_zmin = pml_zmin;
    _kpar_host->pml_zmax = pml_zmax;
    _kpar_host->odx = odx;
    _kpar_host->t = t;
    _kpar_host->src_T = _src_T;
    _kpar_host->src_min = _src_min;
    _kpar_host->src_k = _src_k;

//     dim3 block(16,8);
//     dim3 grid(ceil(_I*_J*_K/128.0/572.0), 572);

    gpuErrchk(cudaMemcpy(_kpar_device,_kpar_host, sizeof(kernelpar), cudaMemcpyHostToDevice));
    kernel_update_E <<< ceil(_I*_J*_K/128.0), 128 >>> (_kpar_device);
//     kernel_update_E <<< grid, 128 >>> (_kpar_device);
//     cudaError_t cudaerr = cudaDeviceSynchronize();
//     printf("update E: kernel launch status \"%s\".\n", cudaGetErrorString(cudaerr));


//    double sum=0;
//    double sumbase=0;
//    unsigned int ui;
//    uint64_t t0 = __builtin_ia32_rdtscp(&ui);
//    uint64_t t1 = __builtin_ia32_rdtscp(&ui);
//    uint64_t elapsed = t1-t0;
//    sumbase += elapsed;
//    t0 = __builtin_ia32_rdtscp(&ui);

//    t1 = __builtin_ia32_rdtscp(&ui);
//    elapsed = t1-t0;
//    sum += elapsed;

//    std::cout << "Base:" << sumbase << std::endl;
//    std::cout << "For-loop:" << sum << std::endl;

}

///////////////////////////////////////////////////////////////////////////
// PML Management
///////////////////////////////////////////////////////////////////////////


void fdtd::FDTD::set_pml_widths(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    _w_pml_x0 = xmin; _w_pml_x1 = xmax;
    _w_pml_y0 = ymin; _w_pml_y1 = ymax;
    _w_pml_z0 = zmin; _w_pml_z1 = zmax;
}

void fdtd::FDTD::set_pml_properties(double sigma, double alpha, double kappa, double pow)
{
    _sigma = sigma;
    _alpha = alpha;
    _kappa = kappa;
    _pow   = pow;

    compute_pml_params();
}

void fdtd::FDTD::build_pml()
{
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,
        zmin = _w_pml_z0, zmax = _Nz-_w_pml_z1;

    // touches xmin boudary
    if(_k0 < xmin) {
        N = _I * _J * (xmin - _k0);

        // Clean up old arrays and allocate new ones
        delete [] _pml_Eyx0; _pml_Eyx0 = NULL;
        delete [] _pml_Ezx0; _pml_Ezx0 = NULL;
        _pml_Eyx0 = new double[N];
        _pml_Ezx0 = new double[N];

        delete [] _pml_Hyx0; _pml_Hyx0 = NULL;
        delete [] _pml_Hzx0; _pml_Hzx0 = NULL;
        _pml_Hyx0 = new double[N];
        _pml_Hzx0 = new double[N];
    }

    // touches xmax boundary
    if(_k0 +_K > xmax) {
        N = _I * _J * (_k0  + _K - xmax);

        // Clean up old arrays and allocate new ones
        delete [] _pml_Eyx1; _pml_Eyx1 = NULL;
        delete [] _pml_Ezx1; _pml_Ezx1 = NULL;
        _pml_Eyx1 = new double[N];
        _pml_Ezx1 = new double[N];

        delete [] _pml_Hyx1; _pml_Hyx1 = NULL;
        delete [] _pml_Hzx1; _pml_Hzx1 = NULL;
        _pml_Hyx1 = new double[N];
        _pml_Hzx1 = new double[N];
    }

    // touches ymin boundary
    if(_j0 < ymin) {
        N = _I * _K * (ymin - _j0);

        delete [] _pml_Exy0; _pml_Exy0 = NULL;
        delete [] _pml_Ezy0; _pml_Ezy0 = NULL;
        _pml_Exy0 = new double[N];
        _pml_Ezy0 = new double[N];

        delete [] _pml_Hxy0; _pml_Hxy0 = NULL;
        delete [] _pml_Hzy0; _pml_Hzy0 = NULL;
        _pml_Hxy0 = new double[N];
        _pml_Hzy0 = new double[N];
    }

    // touches ymax boundary
    if(_j0 + _J > ymax) {
        N = _I * _K * (_j0 + _J - ymax);

        delete [] _pml_Exy1; _pml_Exy1 = NULL;
        delete [] _pml_Ezy1; _pml_Ezy1 = NULL;
        _pml_Exy1 = new double[N];
        _pml_Ezy1 = new double[N];

        delete [] _pml_Hxy1; _pml_Hxy1 = NULL;
        delete [] _pml_Hzy1; _pml_Hzy1 = NULL;
        _pml_Hxy1 = new double[N];
        _pml_Hzy1 = new double[N];
    }

    // touches zmin boundary
    if(_i0 < zmin) {
        N = _J * _K * (zmin - _i0);

        delete [] _pml_Exz0; _pml_Exz0 = NULL;
        delete [] _pml_Eyz0; _pml_Eyz0 = NULL;
        _pml_Exz0 = new double[N];
        _pml_Eyz0 = new double[N];

        delete [] _pml_Hxz0; _pml_Hxz0 = NULL;
        delete [] _pml_Hyz0; _pml_Hyz0 = NULL;
        _pml_Hxz0 = new double[N];
        _pml_Hyz0 = new double[N];
    }

    // touches zmax boundary
    if(_i0 + _I > zmax) {
        N = _J * _K * (_i0 + _I - zmax);

        delete [] _pml_Hxz1; _pml_Hxz1 = NULL;
        delete [] _pml_Hyz1; _pml_Hyz1 = NULL;
        _pml_Exz1 = new double[N];
        _pml_Eyz1 = new double[N];

        delete [] _pml_Hxz1; _pml_Hxz1 = NULL;
        delete [] _pml_Hyz1; _pml_Hyz1 = NULL;
        _pml_Hxz1 = new double[N];
        _pml_Hyz1 = new double[N];
    }

    // (re)compute the spatially-dependent PML parameters
    compute_pml_params();
}

void fdtd::FDTD::reset_pml()
{
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,
        zmin = _w_pml_z0, zmax = _Nz-_w_pml_z1;

    // touches xmin boudary
    if(_k0 < xmin) {
        N = _I * _J * (xmin - _k0);
        std::fill(_pml_Eyx0, _pml_Eyx0 + N, 0);
        std::fill(_pml_Ezx0, _pml_Ezx0 + N, 0);
        std::fill(_pml_Hyx0, _pml_Hyx0 + N, 0);
        std::fill(_pml_Hzx0, _pml_Hzx0 + N, 0);
    }

    // touches xmax boundary
    if(_k0 +_K > xmax) {
        N = _I * _J * (_k0  + _K - xmax);
        std::fill(_pml_Eyx1, _pml_Eyx1 + N, 0);
        std::fill(_pml_Ezx1, _pml_Ezx1 + N, 0);
        std::fill(_pml_Hyx1, _pml_Hyx1 + N, 0);
        std::fill(_pml_Hzx1, _pml_Hzx1 + N, 0);
    }

    // touches ymin boundary
    if(_j0 < ymin) {
        N = _I * _K * (ymin - _j0);
        std::fill(_pml_Exy0, _pml_Exy0 + N, 0);
        std::fill(_pml_Ezy0, _pml_Ezy0 + N, 0);
        std::fill(_pml_Hxy0, _pml_Hxy0 + N, 0);
        std::fill(_pml_Hzy0, _pml_Hzy0 + N, 0);
    }

    // touches ymax boundary
    if(_j0 + _J > ymax) {
        N = _I * _K * (_j0 + _J - ymax);
        std::fill(_pml_Exy1, _pml_Exy1 + N, 0);
        std::fill(_pml_Ezy1, _pml_Ezy1 + N, 0);
        std::fill(_pml_Hxy1, _pml_Hxy1 + N, 0);
        std::fill(_pml_Hzy1, _pml_Hzy1 + N, 0);
    }

    // touches zmin boundary
    if(_i0 < zmin) {
        N = _J * _K * (zmin - _i0);
        std::fill(_pml_Exz0, _pml_Exz0 + N, 0);
        std::fill(_pml_Eyz0, _pml_Eyz0 + N, 0);
        std::fill(_pml_Hxz0, _pml_Hxz0 + N, 0);
        std::fill(_pml_Hyz0, _pml_Hyz0 + N, 0);
    }

    // touches zmax boundary
    if(_i0 + _I > zmax) {
        N = _J * _K * (_i0 + _I - zmax);
        std::fill(_pml_Exz1, _pml_Exz1 + N, 0);
        std::fill(_pml_Eyz1, _pml_Eyz1 + N, 0);
        std::fill(_pml_Hxz1, _pml_Hxz1 + N, 0);
        std::fill(_pml_Hyz1, _pml_Hyz1 + N, 0);
    }

}

void fdtd::FDTD::compute_pml_params()
{
    double pml_dist, pml_factor, sigma, alpha, kappa, b, c;

    // clean up the previous arrays and allocate new ones
    delete [] _kappa_H_x; _kappa_H_x = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _kappa_H_y; _kappa_H_y = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _kappa_H_z; _kappa_H_z = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _kappa_E_x; _kappa_E_x = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _kappa_E_y; _kappa_E_y = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _kappa_E_z; _kappa_E_z = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _bHx; _bHx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _bHy; _bHy = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _bHz; _bHz = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _bEx; _bEx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _bEy; _bEy = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _bEz; _bEz = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _cHx; _cHx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _cHy; _cHy = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _cHz; _cHz = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _cEx; _cEx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _cEy; _cEy = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _cEz; _cEz = new double[_w_pml_z0 + _w_pml_z1];

    // calculate the PML parameters. These parameters are all functions of
    // the distance from the ONSET of the PML edge (which begins in the simulation
    // domain interior.
    // Note: PML parameters are ordered such that distance from PML onset
    // always increases with index.
    
    // setup xmin PML parameters
    for(int k = 0; k < _w_pml_x0; k++) {
        pml_dist = double(k - 0.5)/_w_pml_x0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);
        if(pml_factor < 0) pml_factor = 0;

        // compute H coefficients
        sigma = _sigma * pml_factor;
        alpha = _alpha * (1-pml_factor);
        kappa = (_kappa-1.0) * pml_factor+1.0;
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_x[k] = kappa;
        _bHx[k] = b;
        _cHx[k] = c;

        pml_dist = double(k)/_w_pml_x0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        // compute E coefficients
        sigma = _sigma * pml_factor;
        alpha = _alpha * (1-pml_factor);
        kappa = (_kappa-1) * pml_factor+1;
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_x[k] = kappa;
        _bEx[k] = b;
        _cEx[k] = c;

    }
    for(int k = 0; k < _w_pml_x1; k++) {
        // compute H coefficients
        pml_dist = double(k + 0.5)/_w_pml_x1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_x[_w_pml_x0 + k] = kappa;
        _bHx[_w_pml_x0 + k] = b;
        _cHx[_w_pml_x0 + k] = c;

        //compute E coefficients
        pml_dist = double(k)/_w_pml_x1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_x[_w_pml_x0 + k] = kappa;
        _bEx[_w_pml_x0 + k] = b;
        _cEx[_w_pml_x0 + k] = c;
    }
    for(int j = 0; j < _w_pml_y0; j++) {
        // calc H coefficients
        pml_dist = double(j - 0.5)/_w_pml_y0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);
        if(pml_factor < 0) pml_factor = 0;

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_y[j] = kappa;
        _bHy[j] = b;
        _cHy[j] = c;

        // calc E coefficients
        pml_dist = double(j)/_w_pml_y0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_y[j] = kappa;
        _bEy[j] = b;
        _cEy[j] = c;
    
    }
    for(int j = 0; j < _w_pml_y1; j++) {
         // calc H coeffs
         pml_dist = double(j + 0.5)/_w_pml_y1; // distance from pml edge
         pml_factor = pml_ramp(pml_dist);

         sigma = _sigma * pml_factor;
         kappa = (_kappa-1) * pml_factor+1;
         alpha = _alpha * (1-pml_factor);
         b = exp(-_dt*(sigma/kappa + alpha));
         if(b == 1) c = 0;
         else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_y[_w_pml_y0 + j] = kappa;
        _bHy[_w_pml_y0 + j] = b;
        _cHy[_w_pml_y0 + j] = c;

        // compute E coefficients
        pml_dist = double(j)/_w_pml_y1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha); 

        _kappa_E_y[_w_pml_y0 + j] = kappa;
        _bEy[_w_pml_y0 + j] = b;
        _cEy[_w_pml_y0 + j] = c;
    }

    for(int i = 0; i < _w_pml_z0; i++) {
        // calc H coeffs
        pml_dist = double(i)/_w_pml_z0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c= 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_z[i] = kappa;
        _bHz[i] = b;
        _cHz[i] = c;

        // calc E coeffs
        pml_dist = double(i+0.5)/_w_pml_z0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        // compute coefficients
        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_z[i] = kappa;
        _bEz[i] = b;
        _cEz[i] = c;
    }

    for(int i = 0; i < _w_pml_z1; i++) {
        // calc H coeffs
        pml_dist = double(i)/_w_pml_z1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_z[_w_pml_z0 + i] = kappa;
        _bHz[_w_pml_z0 + i] = b;
        _cHz[_w_pml_z0 + i] = c;

        // calc E coeffs
        pml_dist = double(i - 0.5)/_w_pml_z1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);
        if(pml_factor < 0) pml_factor = 0;

        // compute coefficients
        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_z[_w_pml_z0 + i] = kappa;
        _bEz[_w_pml_z0 + i] = b;
        _cEz[_w_pml_z0 + i] = c;
    }
}

double fdtd::FDTD::pml_ramp(double pml_dist)
{
    return std::pow(pml_dist, _pow);
}

///////////////////////////////////////////////////////////////////////////
// Amp/Phase Calculation management Management
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::set_t0_arrays(complex128 *Ex_t0, complex128 *Ey_t0, complex128 *Ez_t0,
                                complex128 *Hx_t0, complex128 *Hy_t0, complex128 *Hz_t0)
{
    _Ex_t0 = Ex_t0; _Ey_t0 = Ey_t0; _Ez_t0 = Ez_t0;
    _Hx_t0 = Hx_t0; _Hy_t0 = Hy_t0; _Hz_t0 = Hz_t0;
}

void fdtd::FDTD::set_t1_arrays(complex128 *Ex_t1, complex128 *Ey_t1, complex128 *Ez_t1,
complex128 *Hx_t1, complex128 *Hy_t1, complex128 *Hz_t1)
{
_Ex_t1 = Ex_t1; _Ey_t1 = Ey_t1; _Ez_t1 = Ez_t1;
_Hx_t1 = Hx_t1; _Hy_t1 = Hy_t1; _Hz_t1 = Hz_t1;
}

void fdtd::FDTD::capture_pbox_fields(std::complex<double> *Ex_full, std::complex<double> *Ey_full,
                                    std::complex<double> *Ez_full, std::complex<double> *Ex_pbox,
                                    std::complex<double> *Ey_pbox, std::complex<double> *Ez_pbox)
{
    int ind_full, ind_pbox;

    for(int i = _i1; i < _i2; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_full = i*_J*_K + j*_K + k;
                ind_pbox = (i-_i1)*_J*_K + j*_K + k;

                Ex_pbox[ind_pbox] = Ex_full[ind_full];
                Ey_pbox[ind_pbox] = Ey_full[ind_full];
                Ez_pbox[ind_pbox] = Ez_full[ind_full];
            }
        }
    }
}

void fdtd::FDTD::capture_t0_fields()
{
    int ind_local, ind_global;

    for(int i = 0; i < _I; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_local = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_global = i*_J*_K + j*_K + k;

                // Copy the fields at the current time to the auxillary arrays
                _Ex_t0[ind_global] = _Ex[ind_local];
                _Ey_t0[ind_global] = _Ey[ind_local];
                _Ez_t0[ind_global] = _Ez[ind_local];

                _Hx_t0[ind_global] = _Hx[ind_local];
                _Hy_t0[ind_global] = _Hy[ind_local];
                _Hz_t0[ind_global] = _Hz[ind_local];
            }
        }
    }

}

void fdtd::FDTD::capture_t1_fields()
{
    int ind_local, ind_global;

    for(int i = 0; i < _I; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_local = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_global = i*_J*_K + j*_K + k;

                // Copy the fields at the current time to the auxillary arrays
                _Ex_t1[ind_global] = _Ex[ind_local];
                _Ey_t1[ind_global] = _Ey[ind_local];
                _Ez_t1[ind_global] = _Ez[ind_local];

                _Hx_t1[ind_global] = _Hx[ind_local];
                _Hy_t1[ind_global] = _Hy[ind_local];
                _Hz_t1[ind_global] = _Hz[ind_local];
            }
        }
    }

}

void fdtd::FDTD::calc_complex_fields(double t0, double t1)
{
    double f0, f1, phi, A, t0H, t1H;
    int ind_local, ind_global;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;

    for(int i = 0; i < _I; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_local = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_global = i*_J*_K + j*_K + k;
                
                // Compute amplitude and phase for Ex
                // Note: we are careful to assume exp(-i*w*t) time dependence
                f0 = _Ex_t0[ind_global].real;
                f1 = _Ex[ind_local];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ex_t0[ind_global].real = A*cos(phi);
                _Ex_t0[ind_global].imag = -A*sin(phi); 

                // Ey
                f0 = _Ey_t0[ind_global].real;
                f1 = _Ey[ind_local];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ey_t0[ind_global].real = A*cos(phi);
                _Ey_t0[ind_global].imag = -A*sin(phi); 

                // Ez
                f0 = _Ez_t0[ind_global].real;
                f1 = _Ez[ind_local];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ez_t0[ind_global].real = A*cos(phi);
                _Ez_t0[ind_global].imag = -A*sin(phi); 

                // Hx
                f0 = _Hx_t0[ind_global].real;
                f1 = _Hx[ind_local];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hx_t0[ind_global].real = A*cos(phi);
                _Hx_t0[ind_global].imag = -A*sin(phi); 

                // Hy
                f0 = _Hy_t0[ind_global].real;
                f1 = _Hy[ind_local];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hy_t0[ind_global].real = A*cos(phi);
                _Hy_t0[ind_global].imag = -A*sin(phi); 

                // Hz
                f0 = _Hz_t0[ind_global].real;
                f1 = _Hz[ind_local];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hz_t0[ind_global].real = A*cos(phi);
                _Hz_t0[ind_global].imag = -A*sin(phi); 
            }
        }
    }

}


void fdtd::FDTD::calc_complex_fields(double t0, double t1, double t2)
{
    double f0, f1, f2, phi, A, t0H, t1H, t2H;
    int ind_local, ind_global;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;
    t2H = t2 - 0.5*_dt;

    for(int i = 0; i < _I; i++) {
        for(int j = 0; j < _J; j++) {
            for(int k = 0; k < _K; k++) {
                ind_local = (i+1)*(_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
                ind_global = i*_J*_K + j*_K + k;

                // Compute amplitude and phase for Ex
                // Note: we are careful to assume exp(-i*w*t) time dependence
                f0 = _Ex_t0[ind_global].real;
                f1 = _Ex_t1[ind_global].real;
                f2 = _Ex[ind_local];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ex_t0[ind_global].real = A*cos(phi);
                _Ex_t0[ind_global].imag = -A*sin(phi); 

                // Ey
                f0 = _Ey_t0[ind_global].real;
                f1 = _Ey_t1[ind_global].real;
                f2 = _Ey[ind_local];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ey_t0[ind_global].real = A*cos(phi);
                _Ey_t0[ind_global].imag = -A*sin(phi); 

                // Ez
                f0 = _Ez_t0[ind_global].real;
                f1 = _Ez_t1[ind_global].real;
                f2 = _Ez[ind_local];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ez_t0[ind_global].real = A*cos(phi);
                _Ez_t0[ind_global].imag = -A*sin(phi); 

                // Hx
                f0 = _Hx_t0[ind_global].real;
                f1 = _Hx_t1[ind_global].real;
                f2 = _Hx[ind_local];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hx_t0[ind_global].real = A*cos(phi);
                _Hx_t0[ind_global].imag = -A*sin(phi); 

                // Hy
                f0 = _Hy_t0[ind_global].real;
                f1 = _Hy_t1[ind_global].real;
                f2 = _Hy[ind_local];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hy_t0[ind_global].real = A*cos(phi);
                _Hy_t0[ind_global].imag = -A*sin(phi); 

                // Hz
                f0 = _Hz_t0[ind_global].real;
                f1 = _Hz_t1[ind_global].real;
                f2 = _Hz[ind_local];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hz_t0[ind_global].real = A*cos(phi);
                _Hz_t0[ind_global].imag = -A*sin(phi);

            }
        }
    }
}

inline double fdtd::calc_phase(double t0, double t1, double f0, double f1)
{
    if(f0 == 0.0 and f1 == 0) {
        return 0.0;
    }
    else {
        return atan((f1*sin(t0)-f0*sin(t1))/(f0*cos(t1)-f1*cos(t0)));
    }
}

inline double fdtd::calc_amplitude(double t0, double t1, double f0, double f1, double phase)
{
    if(f0*f0 > f1*f1) {
        return f1 / (sin(t1)*cos(phase) + cos(t1)*sin(phase));
    }
    else {
        return f0 / (sin(t0)*cos(phase) + cos(t0)*sin(phase));
    }
}

inline double fdtd::calc_phase(double t0, double t1, double t2, double f0, double f1, double f2)
{
    double f10 = f1 - f0,
           f21 = f2 - f1;

    if(f10 == 0 && f21 == 0) {
        return 0.0;
    }
    else {
        return atan2(f10*(sin(t2)-sin(t1)) - f21*(sin(t1)-sin(t0)), 
                     f21*(cos(t1)-cos(t0)) - f10*(cos(t2)-cos(t1)));
    }
}

inline double fdtd::calc_amplitude(double t0, double t1, double t2, double f0, double f1, double f2, double phase)
{
    double f21 = f2 - f1,
           f10 = f1 - f0;

    if(f21 == 0 && f10 == 0) {
        return 0.0;
    }
    else if(f21*f21 >= f10*f10) {
        return f21 / (cos(phase)*(sin(t2)-sin(t1)) + sin(phase)*(cos(t2)-cos(t1)));
    }
    else {
        return f10 / (cos(phase)*(sin(t1)-sin(t0)) + sin(phase)*(cos(t1)-cos(t0)));
    }
}

///////////////////////////////////////////////////////////////////////////
// Source management
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::add_source(complex128 *Jx, complex128 *Jy, complex128 *Jz,
                            complex128 *Mx, complex128 *My, complex128 *Mz,
                            int i0, int j0, int k0, int I, int J, int K,
                            bool calc_phase)
{
    int ind=0;
    double real, imag;
    SourceArray src = {Jx, Jy, Jz, Mx, My, Mz, i0, j0, k0, I, J, K};

    // these source arrays may *actually* be complex-valued. In the time
    // domain, complex values correspond to temporal phase shifts. We need
    // to convert the complex value to an amplitude and phase. Fortunately,
    // we can use the memory that is already allocated for these values.
    // Specifically, we use src_array.real = amplitude and
    // src_array.imag = phase
    //
    // Important note: EMopt assumes the time dependence is exp(-i*omega*t).
    // In order to account for this minus sign, we need to invert the sign
    // of the calculated phase.
    if(calc_phase) {

    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            for(int k = 0; k < K; k++) {
                ind = i*J*K + j*K + k;

                
                // Jx
                real = Jx[ind].real;
                imag = Jx[ind].imag;

                Jx[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Jx[ind].imag = 0.0;
                else Jx[ind].imag = -1*atan2(imag, real);

                // Jy
                real = Jy[ind].real;
                imag = Jy[ind].imag;

                Jy[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Jy[ind].imag = 0.0;
                else Jy[ind].imag = -1*atan2(imag, real);

                // Jz
                real = Jz[ind].real;
                imag = Jz[ind].imag;

                Jz[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Jz[ind].imag = 0.0;
                else Jz[ind].imag = -1*atan2(imag, real);

                // Mx
                real = Mx[ind].real;
                imag = Mx[ind].imag;

                Mx[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Mx[ind].imag = 0.0;
                else Mx[ind].imag = -1*atan2(imag, real);

                // My
                real = My[ind].real;
                imag = My[ind].imag;

                My[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) My[ind].imag = 0.0;
                else My[ind].imag = -1*atan2(imag, real);

                // Mz
                real = Mz[ind].real;
                imag = Mz[ind].imag;

                Mz[ind].real = sqrt(real*real + imag*imag);
                if(imag == 0 && real == 0) Mz[ind].imag = 0.0;
                else Mz[ind].imag = -1*atan2(imag, real);
                
            }
        }
    }
    }

    _sources.push_back(src);

    _srclen += 1;
    _kpar_host->srclen += 1;
}

void fdtd::FDTD::clear_sources()
{
    _sources.clear();

    _srclen = 0;
    _kpar_host->srclen = 0;
}

void fdtd::FDTD::set_source_properties(double src_T, double src_min)
{
    _src_T = src_T;
    _src_min = src_min;
    _src_k = src_T*src_T / log((1+src_min)/src_min);
    //_src_k = 6.0 / src_T; // rate of src turn on
    //_src_n0 = 1.0 / _src_k * log((1.0-src_min)/src_min); // src delay
}

inline double fdtd::FDTD::src_func_t(int n, double t, double phase)
{
    //return sin(t + phase) / (1.0 + exp(-_src_k*(n-_src_n0)));
    if(t <= _src_T)
        return sin(t + phase)*((1+_src_min) * exp(-(t-_src_T)*(t-_src_T) / _src_k) - _src_min);
    else
        return sin(t + phase);
}

///////////////////////////////////////////////////////////////////////////
// Boundary Conditions
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::set_bc(char* newbc)
{
    for(int i = 0; i < 3; i++){
        _bc[i] = newbc[i];
    }
}

///////////////////////////////////////////////////////////////////////////
// ctypes interface
///////////////////////////////////////////////////////////////////////////

fdtd::FDTD* FDTD_new()
{
    return new fdtd::FDTD();
}

void FDTD_set_wavelength(fdtd::FDTD* fdtd, double wavelength)
{
    fdtd->set_wavelength(wavelength);
}

void FDTD_set_physical_dims(fdtd::FDTD* fdtd, 
                            double X, double Y, double Z,
                            double dx, double dy, double dz)
{
    fdtd->set_physical_dims(X, Y, Z, dx, dy, dz);
}

void FDTD_set_grid_dims(fdtd::FDTD* fdtd, int Nx, int Ny, int Nz)
{
    fdtd->set_grid_dims(Nx, Ny, Nz);
}

void FDTD_set_local_grid(fdtd::FDTD* fdtd, 
                         int k0, int j0, int i0,
                         int K, int J, int I)
{
    fdtd->set_local_grid(k0, j0, i0, K, J, I);
}

void FDTD_set_local_grid_perturb(fdtd::FDTD* fdtd,
                         int i1, int i2)
{
    fdtd->set_local_grid_perturb(i1, i2);
}

void FDTD_set_dt(fdtd::FDTD* fdtd, double dt)
{
    fdtd->set_dt(dt);
}

void FDTD_set_complex_eps(fdtd::FDTD* fdtd, bool complex_eps)
{
    fdtd->set_complex_eps(complex_eps);
}

void FDTD_copyCUDA_field_arrays(fdtd::FDTD* fdtd)
{
    fdtd->copyCUDA_field_arrays();
}

void FDTD_block_CUDA_free(fdtd::FDTD* fdtd)
{
    fdtd->block_CUDA_free();
}

void FDTD_block_CUDA_src_free(fdtd::FDTD* fdtd)
{
    fdtd->block_CUDA_src_free();
}

void FDTD_block_CUDA_malloc_memcpy(fdtd::FDTD* fdtd)
{
    fdtd->block_CUDA_malloc_memcpy();
}

void FDTD_block_CUDA_src_malloc_memcpy(fdtd::FDTD* fdtd)
{
    fdtd->block_CUDA_src_malloc_memcpy();
}

void FDTD_calc_ydAx(fdtd::FDTD* fdtd, size_t size, size_t Nx, size_t Ny, size_t Nz, size_t i0, size_t i1, size_t i2,
                std::complex<double> *ydAx,
                std::complex<double> *Ex_adj, std::complex<double> *Ey_adj, std::complex<double> *Ez_adj,
                std::complex<double> *Ex_fwd, std::complex<double> *Ey_fwd, std::complex<double> *Ez_fwd,
                std::complex<double> *epsx0, std::complex<double> *epsy0, std::complex<double> *epsz0,
                std::complex<double> *epsxp, std::complex<double> *epsyp, std::complex<double> *epszp)
{
    fdtd->calc_ydAx(size, Nx, Ny, Nz, i0, i1, i2, ydAx, Ex_adj, Ey_adj, Ez_adj, Ex_fwd, Ey_fwd, Ez_fwd,
                epsx0, epsy0, epsz0, epsxp, epsyp, epszp);
}

void FDTD_set_field_arrays(fdtd::FDTD* fdtd,
                           double *Ex, double *Ey, double *Ez,
                           double *Hx, double *Hy, double *Hz)
{
    fdtd->set_field_arrays(Ex, Ey, Ez, Hx, Hy, Hz);
}

void FDTD_set_mat_arrays(fdtd::FDTD* fdtd,
                         complex128 *eps_x, complex128 *eps_y, complex128 *eps_z
                         )
{
    fdtd->set_mat_arrays(eps_x, eps_y, eps_z);
}

void FDTD_update_H(fdtd::FDTD* fdtd, int n, double t)
{
    fdtd->update_H(n, t);
}

void FDTD_update_E(fdtd::FDTD* fdtd, int n, double t)
{
    fdtd->update_E(n, t);
}

void FDTD_set_pml_widths(fdtd::FDTD* fdtd, int xmin, int xmax,
                                           int ymin, int ymax,
                                           int zmin, int zmax)
{
    fdtd->set_pml_widths(xmin, xmax, ymin, ymax, zmin, zmax);
}

void FDTD_set_pml_properties(fdtd::FDTD* fdtd, double sigma, double alpha,
                                               double kappa, double pow)
{
    fdtd->set_pml_properties(sigma, alpha, kappa, pow);
}

void FDTD_build_pml(fdtd::FDTD* fdtd)
{
    fdtd->build_pml();
}

void FDTD_reset_pml(fdtd::FDTD* fdtd)
{
    fdtd->reset_pml();
}

void FDTD_set_t0_arrays(fdtd::FDTD* fdtd,
                         complex128 *Ex_t0, complex128 *Ey_t0, complex128 *Ez_t0,
                         complex128 *Hx_t0, complex128 *Hy_t0, complex128 *Hz_t0)
{
    fdtd->set_t0_arrays(Ex_t0, Ey_t0, Ez_t0, Hx_t0, Hy_t0, Hz_t0);
}

void FDTD_set_t1_arrays(fdtd::FDTD* fdtd,
                         complex128 *Ex_t1, complex128 *Ey_t1, complex128 *Ez_t1,
                         complex128 *Hx_t1, complex128 *Hy_t1, complex128 *Hz_t1)
{
    fdtd->set_t1_arrays(Ex_t1, Ey_t1, Ez_t1, Hx_t1, Hy_t1, Hz_t1);
}

double FDTD_calc_phase_2T(double t0, double t1, double f0, double f1)
{
    return fdtd::calc_phase(t0, t1, f0, f1);
}

double FDTD_calc_amplitude_2T(double t0, double t1, double f0, double f1, double phase)
{
    return fdtd::calc_amplitude(t0, t1, f0, f1, phase);
}

double FDTD_calc_phase_3T(double t0, double t1, double t2, double f0, double f1, double f2)
{
    return fdtd::calc_phase(t0, t1, t2, f0, f1, f2);
}

double FDTD_calc_amplitude_3T(double t0, double t1, double t2, double f0, double f1, double f2, double phase)
{
    return fdtd::calc_amplitude(t0, t1, t2, f0, f1, f2, phase);
}

void FDTD_capture_pbox_fields(fdtd::FDTD* fdtd, std::complex<double> *Ex_full, std::complex<double> *Ey_full,
                            std::complex<double> *Ez_full, std::complex<double> *Ex_pbox,
                            std::complex<double> *Ey_pbox, std::complex<double> *Ez_pbox)
{
    fdtd->capture_pbox_fields(Ex_full, Ey_full, Ez_full, Ex_pbox, Ey_pbox, Ez_pbox);
}

void FDTD_capture_t0_fields(fdtd::FDTD* fdtd)
{
    fdtd->capture_t0_fields();
}

void FDTD_capture_t1_fields(fdtd::FDTD* fdtd)
{
    fdtd->capture_t1_fields();
}


void FDTD_calc_complex_fields_2T(fdtd::FDTD* fdtd, double t0, double t1)
{
    fdtd->calc_complex_fields(t0, t1);
}

void FDTD_calc_complex_fields_3T(fdtd::FDTD* fdtd, double t0, double t1, double t2)
{
    fdtd->calc_complex_fields(t0, t1, t2);
}

void FDTD_add_source(fdtd::FDTD* fdtd,
                     complex128 *Jx, complex128 *Jy, complex128 *Jz,
                     complex128 *Mx, complex128 *My, complex128 *Mz,
                     int i0, int j0, int k0, int I, int J, int K, bool calc_phase)
{
    fdtd->add_source(Jx, Jy, Jz, Mx, My, Mz, i0, j0, k0, I, J, K, calc_phase);
}

void FDTD_clear_sources(fdtd::FDTD* fdtd)
{
    fdtd->clear_sources();
}

void FDTD_set_source_properties(fdtd::FDTD* fdtd, double src_T, double src_min)
{
    fdtd->set_source_properties(src_T, src_min);
}

double FDTD_src_func_t(fdtd::FDTD* fdtd, int n, double t, double phase)
{
    return fdtd->src_func_t(n, t, phase);
}

void FDTD_set_bc(fdtd::FDTD* fdtd, char* newbc)
{
    fdtd->set_bc(newbc);
}

// Ghost communication helper functions
void FDTD_copy_to_ghost_comm(double* src, complex128* ghost, int I, int J, int K)
{
    unsigned int nstart = 0,
                 ind_ijk, ind_ghost;

    // copy xmin
    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            ind_ijk = (i+1)*(J+2)*(K+2) + (j+1)*(K+2) + 1;
            ind_ghost = nstart + j + i*J;

            ghost[ind_ghost] = src[ind_ijk];
        }
    }

    // copy xmax
    nstart = I*J;
    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            ind_ijk = (i+1)*(J+2)*(K+2) + (j+1)*(K+2) + K;
            ind_ghost = nstart + j + i*J;

            ghost[ind_ghost] = src[ind_ijk];
        }
    }

    // copy ymin
    nstart = 2*I*J;
    for(int i = 0; i < I; i++) {
        for(int k = 0; k < K; k++) {
            ind_ijk = (i+1)*(J+2)*(K+2) + 1*(K+2) + k + 1;
            ind_ghost = nstart + k + i*K;

            ghost[ind_ghost] = src[ind_ijk];
        }
    }

    // copy ymax
    nstart = 2*I*J + I*K;
    for(int i = 0; i < I; i++) {
        for(int k = 0; k < K; k++) {
            ind_ijk = (i+1)*(J+2)*(K+2) + J*(K+2) + k + 1;
            ind_ghost = nstart + k + i*K;

            ghost[ind_ghost] = src[ind_ijk];
        }
    }

    // copy zmin
    nstart = 2*I*J + 2*I*K;
    for(int j = 0; j < J; j++) {
        for(int k = 0; k < K; k++) {
            ind_ijk = 1*(J+2)*(K+2) + (j+1)*(K+2) + k + 1;
            ind_ghost = nstart + k + j*K;

            ghost[ind_ghost] = src[ind_ijk];
        }
    }

    // copy zmax
    nstart = 2*I*J + 2*I*K + J*K;
    for(int j = 0; j < J; j++) {
        for(int k = 0; k < K; k++) {
            ind_ijk = I*(J+2)*(K+2) + (j+1)*(K+2) + k + 1;
            ind_ghost = nstart + k + j*K;

            ghost[ind_ghost] = src[ind_ijk];
        }
    }
}

void FDTD_copy_from_ghost_comm(double* dest, complex128* ghost, int I, int J, int K)
{
    unsigned int nstart = 2*I*J + 2*I*K + 2*J*K,
                 ind_ijk, ind_ghost;

    // copy xmin
    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            ind_ijk = (i+1)*(J+2)*(K+2) + (j+1)*(K+2) + 0;
            ind_ghost = nstart + j + i*J;

            dest[ind_ijk] = ghost[ind_ghost].real;
        }
    }

    // copy xmax
    nstart = 2*I*J + 2*I*K + 2*J*K + I*J;
    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            ind_ijk = (i+1)*(J+2)*(K+2) + (j+1)*(K+2) + K+1;
            ind_ghost = nstart + j + i*J;

            dest[ind_ijk] = ghost[ind_ghost].real;
        }
    }

    // copy ymin
    nstart = 2*I*J + 2*I*K + 2*J*K + 2*I*J;
    for(int i = 0; i < I; i++) {
        for(int k = 0; k < K; k++) {
            ind_ijk = (i+1)*(J+2)*(K+2) + 0*(K+2) + k + 1;
            ind_ghost = nstart + k + i*K;

            dest[ind_ijk] = ghost[ind_ghost].real;
        }
    }

    // copy ymax
    nstart = 2*I*J + 2*I*K + 2*J*K + 2*I*J + I*K;
    for(int i = 0; i < I; i++) {
        for(int k = 0; k < K; k++) {
            ind_ijk = (i+1)*(J+2)*(K+2) + (J+1)*(K+2) + k + 1;
            ind_ghost = nstart + k + i*K;

            dest[ind_ijk] = ghost[ind_ghost].real;
        }
    }

    // copy zmin
    nstart = 2*I*J + 2*I*K + 2*J*K + 2*I*J + 2*I*K;
    for(int j = 0; j < J; j++) {
        for(int k = 0; k < K; k++) {
            ind_ijk = 0*(J+2)*(K+2) + (j+1)*(K+2) + k + 1;
            ind_ghost = nstart + k + j*K;

            dest[ind_ijk] = ghost[ind_ghost].real;
        }
    }

    // copy zmax
    nstart = 2*I*J + 2*I*K + 2*J*K + 2*I*J + 2*I*K + J*K;
    for(int j = 0; j < J; j++) {
        for(int k = 0; k < K; k++) {
            ind_ijk = (I+1)*(J+2)*(K+2) + (j+1)*(K+2) + k + 1;
            ind_ghost = nstart + k + j*K;

            dest[ind_ijk] = ghost[ind_ghost].real;
        }
    }
}
