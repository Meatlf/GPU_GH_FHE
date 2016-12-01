#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
//#include <gmp.h>
#include <vector>
#include <stack>
#include <NTL/tools.h>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <cuda.h>
#include <cuda_runtime.h>

NTL_CLIENT

#include "fhe.h"
#include "randomStuff.h"
#include <new>

#include "timing_dumper.c"
#include "baseCode.c"

   // Utility functions to compute carry bits and add numbers in binay
   //typedef std::stack<ZZ> ZZstack;
   //static void evalSymPolys(vec_ZZ& out, ZZstack& vars, long deg, const ZZ& M);
   //static void gradeSchoolAdd(FHEctxt& c, const mat_ZZ& vars, const ZZ& M);

#define $GPU(call) if((call)!=0) { printf("\nCall \"" #call "\" failed from %s, line %d\n", __FILE__, __LINE__); exit(1); }

extern "C" int getFreeMemory() ;

extern "C" void gpu_set_block_count(int count);

extern "C" void gpuCudaTrans(uint32_t *a, uint32_t *b, uint32_t words);
extern "C" void gpuCudaAdd(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t words, uint32_t *carry);
extern "C" void gpuCudaSub(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t words, uint32_t *carry);
extern "C" void gpuCudaDivQ(uint32_t *r, uint32_t *x, uint32_t xLength, uint32_t qNum);
extern "C" void gpuCudaAddMod(uint32_t *a, uint32_t *b, uint32_t *d, uint32_t dLength, uint32_t *c, uint32_t *carry, uint32_t *c_2, uint32_t *carry_2);

extern "C" void gpuCudaAddNoMod(uint32_t *a, uint32_t *b, uint32_t *c,uint32_t Length,  uint32_t *carry);


extern "C" void gpuCudaSubMod(uint32_t *a, uint32_t *b, uint32_t *d, uint32_t dLength, uint32_t words, uint32_t *c, uint32_t *carry, uint32_t *c_2);
extern "C" void gpuCudaCmpSub(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t words, uint32_t *carry, uint32_t dLength);

extern "C" void gpu_step2_ld2(uint32_t *x, uint32_t xLength, uint2 *X, uint32_t *y, uint32_t yLength, uint2 *Y, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_step1_ld2(uint32_t *x, uint32_t xLength, uint2 *X, uint32_t *y, uint32_t yLength, uint2 *Y, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_step1_ld3(uint32_t *x, uint32_t xLength, uint2 *X, uint32_t *y, uint32_t yLength, uint2 *Y, uint32_t size, uint32_t *outTimes);


extern "C" void gpu_step1_ld3_onlyX(uint32_t *x, uint32_t xLength, uint2 *X, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_step1_ld3_preX(uint32_t *x, uint32_t xLength, uint2 *X, uint32_t *y, uint32_t yLength, uint2 *Y, uint32_t size, uint32_t *outTimes);

extern "C" void gpu_step1(uint2 *x, uint2 *X, uint2 *y, uint2 *Y, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_step0_twid(uint2 *x, uint2 *X, uint2 *y, uint2 *Y, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_step0_twid_onlyX(uint2 *x, uint2 *X, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_step0_twid_preX(uint2 *x, uint2 *X, uint2 *y, uint2 *Y, uint32_t size, uint32_t *outTimes);


extern "C" void gpu_step0_fft8_istep(uint2 *x, uint2 *y, uint2 *Y, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft8(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft8_onlyX(uint2 *x,  uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft16(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft16_onlyX(uint2 *x,  uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft32(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft32_onlyX(uint2 *x,  uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft32_preX(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft8_onlyMult(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft8_onlyIFFT(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft8_onlyAdd(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft16_onlyMult(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft16_onlyIFFT(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft16_onlyAdd(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft32_onlyMult(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft32_onlyIFFT(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft32_onlyAdd(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft64_onlyX(uint2 *x,  uint2 *Z, uint32_t size, uint32_t *outTimes);

extern "C" void gpu_fft64_onlyMult(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft64_onlyIFFT(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft64_onlyAdd(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);


extern "C" void gpu_fft64(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft128(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_fft256(uint2 *x, uint2 *y, uint2 *Z, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_istep2_twid(uint2 *x, uint2 *X, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_istep1_twid(uint2 *x, uint2 *X, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_istep1(uint2 *x, uint2 *X, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_istep0_st2(uint2 *x, uint2 *X, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_istep0_st3(uint2 *x, uint2 *X, uint32_t size, uint32_t *outTimes);
extern "C" void gpu_carry_by2(uint32_t *in, uint32_t size, uint32_t *carries, uint32_t *out, uint32_t length, uint32_t *outTimes);
extern "C" void gpu_resolve_by2(uint32_t size, uint32_t *carries, uint32_t *inOut, uint32_t length, uint32_t *outTimes);
extern "C" void gpu_carry_by3(uint32_t *in, uint32_t size, uint32_t *carries, uint32_t *out, uint32_t length, uint32_t *outTimes);
extern "C" void gpu_resolve_by3(uint32_t size, uint32_t *carries, uint32_t *inOut, uint32_t length, uint32_t *outTimes);

static uint32_t *gpuAlloc32(uint32_t length) {
   uint32_t *remote;

   $GPU(cudaMalloc((void **)&remote, sizeof(uint32_t)*length));
   $GPU(cudaMemset(remote, 0, sizeof(uint32_t)*length));
   return remote;
}

static uint32_t *gpuPush32(uint32_t *data, uint32_t length) {
   uint32_t *remote;
   int       index;

   $GPU(cudaMalloc((void **)&remote, sizeof(uint32_t)*length));
   $GPU(cudaMemcpy(remote, data, sizeof(uint32_t)*length, cudaMemcpyHostToDevice));
   return remote;
}

static uint32_t *gpuPull32(uint32_t *gpu, uint32_t length) {
   uint32_t *result;
   int       index;

   result=(uint32_t *)malloc(sizeof(uint32_t)*length);
   $GPU(cudaMemcpy(result, gpu, sizeof(uint32_t)*length, cudaMemcpyDeviceToHost));
   return result;
}

static void gpuCompare32(uint32_t *correct, uint32_t *gpu, uint32_t size) {
   uint32_t  *local;
   int        index, count;

   local=(uint32_t *)malloc(size*sizeof(uint32_t));
   $GPU(cudaMemcpy(local, gpu, size*sizeof(uint32_t), cudaMemcpyDeviceToHost));

   printf("Comparing uint32 data:\n");
   for(index=0;index<size;index++) {
      if(local[index]!=correct[index]) {
         printf("%04d  %08X %08X\n", index, correct[index], local[index]);
         if(++count==50)
            break;
      }
   }
   if(index<size)
      printf("Compare aborted...\n");
   else
      printf("Compare done...\n");
   free(local);
}

static uint2 *gpuAlloc64(uint32_t length) {
   uint2 *remote;

   $GPU(cudaMalloc((void **)&remote, sizeof(uint2)*length));
   $GPU(cudaMemset(remote, 0, sizeof(uint2)*length));
   return remote;
}

static uint2 *gpuPush64(uint64_t *data, uint32_t length) {
   uint2 *remote, *local;
   int    index;

   local=(uint2 *)malloc(sizeof(uint2)*length);
   for(index=0;index<length;index++)
      local[index]=make_uint2(data[index] & 0xFFFFFFFF, data[index]>>32);

   $GPU(cudaMalloc((void **)&remote, sizeof(uint2)*length));
   $GPU(cudaMemcpy(remote, local, sizeof(uint2)*length, cudaMemcpyHostToDevice));
   free(local);
   return remote;
}

static uint64_t *gpuPull64(uint2 *gpu, uint32_t length) {
   uint2    *local;
   uint64_t *result;
   int       index;

   result=(uint64_t *)malloc(sizeof(uint64_t)*length);
   local=(uint2 *)malloc(sizeof(uint2)*length);
   $GPU(cudaMemcpy(local, gpu, sizeof(uint2)*length, cudaMemcpyDeviceToHost));
   for(index=0;index<length;index++) {
      result[index]=local[index].y;
      result[index]=(result[index]<<32) + local[index].x;
   }
   free(local);
   return result;
}

static void gpuCompare64(uint64_t *correct, uint2 *gpu, uint32_t size) {
   uint2    *local;
   uint64_t  convert;
   int       index, count=0;

   local=(uint2 *)malloc(size*sizeof(uint2));
   $GPU(cudaMemcpy(local, gpu, size*sizeof(uint2), cudaMemcpyDeviceToHost));

   printf("Comparing uint2 data:\n");
   for(index=0;index<size;index++) {
      convert=local[index].y;
      convert=(convert<<32)+local[index].x;
      if(_normalize(convert)!=_normalize(correct[index])) {
         printf("%04d  %016lX %016lX\n", index, _normalize(correct[index]), _normalize(convert));
         if(++count==50)
            break;
      }
   }
   if(index<size)
      printf("Compare aborted...\n");
   else
      printf("Compare done...\n");
   free(local);
}

static void gpuFree(void *chunk) {
   $GPU(cudaFree(chunk));
}

static void gpuDumpTimes(uint32_t blocks, uint32_t kernels, uint32_t *times) {
   uint32_t *local=gpuPull32(times, blocks*kernels*64);


   dumpTimes(blocks, kernels, local);
   free(local);
}



/* Takes two integer 0<=n<d, and returns an nBits-bit integer, with
* the top bit viewed as the bit to the left of the binary point,
* and the lower bits are viewed as being to the right of the point.
* 
* The returned value is the binary representation of the rational
* number n/d, rounded to precision nBits-1.
********************************************************************/
static inline unsigned long getBinaryRep(ZZ n, const ZZ& d, long nBits)
   // It is assumed that nBits fit in one long integer.
{
   // (n * 2^nBits)/d gives nBits bits of precision (one more than needed)
   n <<= nBits;
   n /= d;         // integer division implies truncation

   unsigned long sn = to_long(n); // a single precision variant
   sn = (sn >> 1) + (sn & 1);     // one less bit, but round instead of truncate
   // NOTE: the addition of (sn&1) could make sn as large as 2^{nBits}. For
   // this case we need to remember the bit to the left of the binary point.

   return sn;
}


void gpuMultiply(uint32_t *gpuX, uint32_t xLength, uint32_t *gpuY, uint32_t yLength, uint32_t *gpuZ, uint32_t samples) {
   uint32_t     *times;
   uint2       *gpuA, *gpuB, *gpuC, *gpuD, *gpuE, *gpuF, *gpuOut;


   times=gpuAlloc32(5*64*64);
   gpuA=gpuAlloc64(samples);
   gpuB=gpuAlloc64(samples);
   gpuC=gpuAlloc64(samples);
   gpuD=gpuAlloc64(samples);
   gpuE=gpuAlloc64(samples);
   gpuF=gpuAlloc64(samples);


   if(samples==32*1024) {
      gpu_step1_ld3(gpuX, xLength, gpuA, gpuY, yLength, gpuB, samples, NULL);
      gpu_step0_fft8_istep(gpuA, gpuB, gpuE, samples, NULL);
      gpu_istep0_st3(gpuE, gpuF, samples, NULL);
      gpuOut=gpuF;
   }

   if(samples==64*1024) {
      gpu_set_block_count(64);
      gpu_step1_ld3(gpuX, xLength, gpuA, gpuY, yLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft16(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st3(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
   }

   if(samples==128*1024) {
      gpu_set_block_count(64);
      gpu_step1_ld3(gpuX, xLength, gpuA, gpuY, yLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st3(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
   }

   if(samples==256*1024) {
      gpu_set_block_count(64);
      gpu_step1_ld2(gpuX, xLength, gpuA, gpuY, yLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft64(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
   }

   if(samples==512*1024) {
      gpu_step1_ld2(gpuX, xLength, gpuA, gpuY, yLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft128(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
   }

   if(samples==1024*1024) {
      gpu_step1_ld2(gpuX, xLength, gpuA, gpuY, yLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft256(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
   }

   if(samples==2048*1024) {

      gpu_step2_ld2(gpuX, xLength, gpuA, gpuY, yLength, gpuB, samples, NULL);
      gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft8_onlyX(gpuC, gpuA, samples, NULL);
        
      gpu_fft8_onlyX(gpuD, gpuB, samples, NULL);
      
      gpu_fft8_onlyMult(gpuA,NULL, gpuB, samples, NULL);

      gpu_fft8_onlyIFFT(gpuB,NULL,gpuE,samples,NULL);
      
      gpu_istep1(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
   }

   if(samples==4096*1024) {
      gpu_step2_ld2(gpuX, xLength, gpuA, gpuY, yLength, gpuB, samples, NULL);
      gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_step0_twid(gpuC, gpuA, gpuD, gpuB, samples, NULL);

      gpu_fft16(gpuA, gpuB, gpuE, samples, NULL);
      /*gpu_fft16_onlyX(gpuA, gpuC, samples, NULL);
      gpu_fft16_onlyX(gpuB, gpuD, samples, NULL);
      gpu_fft16_onlyMult(gpuC, NULL, gpuD, samples, NULL);
      gpu_fft16_onlyIFFT(gpuD,NULL,gpuE,samples,NULL);*/
      gpu_istep2_twid(gpuE, gpuF, samples, NULL);
      gpu_istep1(gpuF, gpuE, samples, NULL);
      gpu_istep0_st2(gpuE, gpuF, samples, NULL);
      gpuOut=gpuF;
   }

   if(samples<=128*1024) {
      gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength+yLength, NULL);
      gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuZ, xLength+yLength, NULL);
   }
   else {
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength+yLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, xLength+yLength, NULL);
   }


   $GPU(cudaFree(gpuA));
   $GPU(cudaFree(gpuB));
   $GPU(cudaFree(gpuC));
   $GPU(cudaFree(gpuD));
   $GPU(cudaFree(gpuE));
   $GPU(cudaFree(gpuF));
}

void Multiply(uint32_t *x, uint32_t xLength, uint32_t *y, uint32_t yLength, uint32_t *z, int samples) {
   uint32_t    *gpuX, *gpuY, *gpuZ;
   int          bits;

   bits=samples/4*32;

   gpuX=gpuPush32(x, xLength);
   gpuY=gpuPush32(y, yLength);
   gpuZ=gpuAlloc32(xLength+yLength);

   gpuMultiply(gpuX, xLength, gpuY, yLength, gpuZ, samples);

   //   gpuDumpTimes(64, 5, times);
   $GPU(cudaMemcpy(z, gpuZ, sizeof(uint32_t)*(unsigned int64_t)(xLength+yLength), cudaMemcpyDeviceToHost));

   $GPU(cudaFree(gpuX));
   $GPU(cudaFree(gpuY));
   $GPU(cudaFree(gpuZ));
}

void testMultiply()
{
   uint32_t* a = (uint32_t*)malloc(4096*1024);
   uint32_t* b = (uint32_t*)malloc(4096*1024);
   uint32_t* c = (uint32_t*)malloc(4096*1024*2);

   memset(a,0,4096*1024);
   a[0] = 50;
   b[0] = 20;
   memset(c,0,4096*1024*2);
   /*Multiply(a,4096*1024/4 , b, 4096*1024/4,c,4096*1024);
   printf("test mul 4096 = %u...%u\n",c[4096*1024*2/4-1],c[0]);
   
   for (int i = 0; i < 4096*1024/4; i ++)
   {
      if (c[i]!=b[i])
      {
         printf("c = %u, b = %u, in %d\n",c[i],b[i],i);
         getchar();
      }
   }

   memset(c,0,4096*1024*2);*/
   Multiply(a,1024*1024/4 , b, 1024*1024/4,c,2048*1024);
   printf("test mul 2048 = %d\n",c[0]);
   for (int i = 0; i < 20; i++)printf("%d",c[i]);
   free(a);
   free(b);
   free(c);
}

//********************GPU codes for ProcessBlock*********************/
//This code has the same precedure with Gentry's code, you can refer to
//Genry's code to understand it step by step.
// Processing the i'th public-key block:
// Compute encryption of top p bits of \sum_j sigma_j*(c*x*R^j mod det)/det
// + encryption of top bit (where the LSBs are XORed in)
void inline FHEkeys::gpuProcessBlock(uint32_t *gpuVars, uint2 **gpuCtxtsFFT, const FHEctxt& c, long i, ZZ& temp) const
{

   uint32_t     *c_int;
   c_int=(uint32_t *)malloc(sizeof(uint32_t)*words);

   uint32_t   *gpuCarry;
   uint32_t   *gpuCarry_2;
   uint32_t   *gpuC_2;
   uint2   *gpuPsums;
   uint2 *gpuVars1;

   gpuCarry = gpuAlloc32(2*words);
   gpuCarry_2 = gpuAlloc32(2*words);
   gpuC_2 = gpuAlloc32(words);
   gpuPsums = gpuAlloc64((samples)*(prms.p+1)); 
   gpuVars1 = gpuAlloc64((samples)*(prms.p+1)); 

   uint32_t* gpuStp;
   uint32_t    *gpuT, *gpuTU, *gpuS, *gpuZ;         //Barret Multiplication
   uint32_t    *gpuR;
   uint2       *gpuA, *gpuB, *gpuC, *gpuD, *gpuE, *gpuF, *gpuOut;
   uint32_t    tLength, nLength, xLength, uLength, tuLength, sLength, zLength;
   tLength = 2*words;
   uLength = words;
   nLength = words;
   xLength = words;
   tuLength = tLength + uLength;
   sLength = tuLength - q;
   zLength = sLength + nLength;

   gpuT=gpuAlloc32(tLength);          //Barret Multiplication
   gpuTU=gpuAlloc32(tuLength);
   gpuS=gpuAlloc32(sLength);
   gpuZ=gpuAlloc32(zLength);
   gpuR=gpuAlloc32(zLength);
   gpuA=gpuAlloc64(samples*2);
   gpuB=gpuAlloc64(samples*2);
   gpuC=gpuAlloc64(samples*2);
   gpuD=gpuAlloc64(samples*2);
   gpuE=gpuAlloc64(samples*2);
   gpuF=gpuAlloc64(samples*2); 

   long nCtxts = mChoose2(prms.S);
   unsigned long baseIdx = i * nCtxts;

   int k;

   unsigned long j, j1, j2;
   ZZ factor = pkBlocks[i].x;

   uint32_t   *tempfactor1,*tempfactor2,*tempc;

   tempfactor1 = gpuAlloc32(words);
   tempfactor2 = gpuAlloc32(words);
   tempc = gpuAlloc32(words);

   uint32_t   *cputempfactor1,*cputempfactor2,*cputempc;

   cputempfactor1 = (uint32_t *)malloc(sizeof(uint32_t)*words);
   cputempfactor2 = (uint32_t *)malloc(sizeof(uint32_t)*words);
   cputempc = (uint32_t *)malloc(sizeof(uint32_t)*words);

   BytesFromZZ((unsigned char *)cputempfactor1, factor, sizeof(uint32_t)*words);
   BytesFromZZ((unsigned char *)cputempc, c, sizeof(uint32_t)*words);

   cudaMemcpy(tempfactor1,  cputempfactor1, sizeof(uint32_t)*words, cudaMemcpyHostToDevice);
   cudaMemcpy(tempc,  cputempc, sizeof(uint32_t)*words, cudaMemcpyHostToDevice);

   //theBarrettEng.gpuBarrett(tempfactor2,tempfactor1,tempc);
   gpu_set_block_count(64);

   if(samples<=128*1024)
   {
      gpu_step1_ld3_onlyX(tempc, words, gpuB, samples, NULL);
      gpu_step0_twid_onlyX(gpuB, gpuD,  samples, NULL);
      gpu_fft32_onlyX(gpuD,gpuE,samples,NULL);
   }
   else if(samples<=512*1024)
   {

      gpu_step1_ld2(tempc, words, gpuB,tempc, words, gpuD, samples, NULL);
      gpu_step0_twid_onlyX(gpuB, gpuD,  samples, NULL);
      gpu_fft64_onlyX(gpuD,gpuE,samples,NULL);
   }
   else 
   {
      gpu_step2_ld2(tempc, words, gpuB,tempc, words, gpuD, samples, NULL);
      gpu_step1(gpuB, gpuD,gpuA, gpuC,  samples, NULL);
      gpu_fft8_onlyX(gpuD,gpuE,samples,NULL);
   }

   if(samples <= 128*1024)
   {
      gpu_fft32_onlyMult((uint2*)prex[i],NULL,gpuE,samples,NULL);
      gpu_fft32_onlyIFFT(gpuE,NULL,gpuE,samples,NULL);
   }
   else if(samples <= 512*1024)
   {
      gpu_fft64_onlyMult((uint2*)prex[i],NULL,gpuE,samples,NULL);
      gpu_fft64_onlyIFFT(gpuE,NULL,gpuE,samples,NULL);

   }
   else 
      {
      gpu_fft8_onlyMult((uint2*)prex[i],NULL,gpuE,samples,NULL);
      gpu_fft8_onlyIFFT(gpuE,NULL,gpuD,samples,NULL);
   }

   if(samples<=128*1024){
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st3(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
      gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
   }
   else if(samples<=512*1024)
   {
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
   }
   else
   {
      gpu_istep1(gpuD, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);

   }


   if(samples<=128*1024){
      //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
      gpu_set_block_count(64);
      gpu_step1_ld3(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st3(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
      gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

      //   gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit
      gpuStp = gpuTU+q;
      //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
      gpu_set_block_count(64);
      gpu_step1_ld3(gpuStp, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st3(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
      gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);

      gpuCudaSub (gpuT, gpuZ, gpuR, xLength, gpuCarry);          //R < -- T-Z  2*n-bit   
      gpuCudaCmpSub (gpuR, gpuN, tempfactor2 , xLength, gpuCarry, dLength);          //

   }
   else if(samples<=512*1024)
   {
      //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
      gpu_set_block_count(64);
      gpu_step1_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples*2, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples*2, NULL);
      gpu_fft128(gpuC, gpuD, gpuE, samples*2, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples*2, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples*2, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
      gpu_resolve_by2(samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
      //   gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit
      gpuStp = gpuTU+q;

      //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
      gpu_set_block_count(64);
      gpu_step1_ld2(gpuStp, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft64(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);

      gpuCudaSub (gpuT, gpuZ, gpuR, xLength, gpuCarry);          //R < -- T-Z  2*n-bit   
      gpuCudaCmpSub (gpuR, gpuN, tempfactor2 , xLength, gpuCarry, dLength);          //

   }
   else
   {
      uint32_t* cput = gpuPull32(gpuT,tLength);
         uint32_t* cpuu = gpuPull32(gpuU,uLength);
         ZZ zt = ZZFromBytes((unsigned char*)cput,tLength*4);
         ZZ ut = ZZFromBytes((unsigned char*)cpuu,uLength*4);
         zt *= ut;
         zt>>=(q*32);
         BytesFromZZ((unsigned char*)cpuu,zt,uLength*4);
         cudaMemcpy(gpuTU,(unsigned char*)cpuu,uLength*4,cudaMemcpyHostToDevice);
         free(cput);
         free(cpuu);

      //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
      /*gpu_set_block_count(64);
      gpu_step1_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples*2, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples*2, NULL);
      gpu_fft128(gpuC, gpuD, gpuE, samples*2, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples*2, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples*2, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
      gpu_resolve_by2(samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);*/

      //   gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit

      gpuStp = gpuTU+q;
      //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
      gpu_set_block_count(64);
      gpu_step2_ld2(gpuTU, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
      gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_step0_fft8_istep(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);

      gpuCudaSub (gpuT, gpuZ, gpuR, xLength, gpuCarry);          //R < -- T-Z  2*n-bit   
      gpuCudaCmpSub (gpuR, gpuN, tempfactor2 , xLength, gpuCarry, dLength);          //

   }

   cudaMemcpy(cputempfactor2,  tempfactor2, sizeof(uint32_t)*words, cudaMemcpyDeviceToHost);
   factor = ZZFromBytes((const unsigned char *)cputempfactor2,sizeof(uint32_t)*words);

   free(cputempfactor1);
   free(cputempfactor2);
   free(cputempc);

   $GPU(cudaFree(tempfactor1));
   $GPU(cudaFree(tempfactor2));   
   $GPU(cudaFree(tempc));
   //#define DETAIL_TIMING
#ifdef DETAIL_TIMING
   float gputime = 0;
   float gputimeall = 0;
   float gpumultime = 0;
   cudaEvent_t start, stop; 
   cudaEventCreate(&start); 
   cudaEventCreate(&stop); 

   int numadds = 0;
   int nummuls = 0;

   double tptime =-GetTime();
   unsigned long volatile binary;
   for (int i = 0; i < 512; i ++)

   {
      factor <<= prms.logR;
      factor %= det;
      binary = getBinaryRep(factor, det, prms.p+1);

   }

   tptime +=GetTime();

   cout<<"cx time="<<tptime<<endl;

#endif

   uint2* gpuCtxtsFFTtmp[nCtxts];
   if(samples>128*1024){
      for(int j = 0; j < nCtxts; j ++)
      {
         if(samples <= 512*1024){
            $GPU(cudaMalloc((void **)&gpuCtxtsFFTtmp[j], sizeof(uint2)*samples));
            $GPU(cudaMemcpy(gpuCtxtsFFTtmp[j], gpuCtxtsFFT[baseIdx + j], sizeof(uint2)*samples, cudaMemcpyHostToDevice));
         }   
      }

   }
   else
   {
      for(int j = 0; j < nCtxts; j ++)
      {
         gpuCtxtsFFTtmp[j] = gpuCtxtsFFT[baseIdx + j];
      }
   }


   cudaMemset(gpuVars1, 0, sizeof(uint2)*samples*(prms.p+1));

   for (j=j1=0; j1<nCtxts-1; j1++) {       // sk-bits indexed by (j1,*) pairs  nCtxts-1

      cudaMemset(gpuPsums, 0, sizeof(uint2)*samples*(prms.p+1));

      //cudaMemset(gpuPsums1, 0, sizeof(uint32_t)*words*(prms.p+1));

      for (j2=j1+1; j2<nCtxts; j2++) {
         // get the top bits of factor/det. The code below assumes
         // that p+1 bits can fit in one unsigned long
         //unsigned long binary = binaries[j];

         unsigned long binary = getBinaryRep(factor, det, prms.p+1);

         if (IsOdd(factor))     // "xor" the LSB to column 0
            binary ^= (1UL << prms.p);

         // For every 1 bit, add the current ciphertext to the partial sums

         if(samples > 512*1024){
            //cout<<"alloc j2="<<j2<<endl;
            $GPU(cudaMalloc((void **)&gpuCtxtsFFTtmp[j2], sizeof(uint2)*samples));
            $GPU(cudaMemcpy(gpuCtxtsFFTtmp[j2], gpuCtxtsFFT[baseIdx + j2], sizeof(uint2)*samples, cudaMemcpyHostToDevice));
         }

         for (k=0; k<prms.p+1; k++)
         {
            if (bit(binary, k) == 1) {

               long k2;
               k2 = prms.p+1 -k-1;

#ifdef DETAIL_TIMING
               cudaEventRecord(start,0); 
#endif
               if(samples <= 128*1024)
               {
                  gpu_fft32_onlyAdd(gpuCtxtsFFTtmp[(j2)],NULL,&gpuPsums[k2*samples],samples,NULL);
               }
               else if(samples <= 512*1024)
               {
                  gpu_fft64_onlyAdd(gpuCtxtsFFTtmp[(j2)],NULL,&gpuPsums[k2*samples],samples,NULL);
               }
               else
               {
                  gpu_fft8_onlyAdd(gpuCtxtsFFTtmp[(j2)],NULL,&gpuPsums[k2*samples],samples,NULL);
               }
#ifdef DETAIL_TIMING
               numadds ++;
               cudaEventRecord(stop,0);
               cudaEventSynchronize(stop);
               cudaEventElapsedTime(&gputime,start,stop);

               gputimeall+=gputime;
#endif

            }
         }

         if(samples > 512*1024){
            cudaFree(gpuCtxtsFFTtmp[j2]);
         }


         //gpuCudaAddMod(&gpuCtxts[0*words], &gpuCtxts[1*words], gpuN, dLength, &gpuCtxts[2*words], gpuCarry, gpuC_2);
         //gpuMultiply(&gpuCtxts[0*words], words/2, &gpuCtxts[1*words], words/2, gpuT, samples);

         j++;              // done with this element
         if (j < prms.S) { // compute next element = current * R mod det
            factor <<= prms.logR;
            factor %= det;
         }
         else break;       // don't add more than S elements
      }

      // multiply partial sums by ctxts[j1], then add to sum


      if(samples > 512*1024){
         $GPU(cudaMalloc((void **)&gpuCtxtsFFTtmp[j1], sizeof(uint2)*samples));
         $GPU(cudaMemcpy(gpuCtxtsFFTtmp[j1], gpuCtxtsFFT[baseIdx + j1], sizeof(uint2)*samples, cudaMemcpyHostToDevice));
      }


      for (k=0; k<prms.p+1; k++) {
         //MulMod(psums[k], psums[k], ctxts[baseIdx+j1], det);
#ifdef DETAIL_TIMING
         cudaEventRecord(start,0);
#endif



         if(samples <= 128*1024)
         {
            gpu_fft32_onlyMult(gpuCtxtsFFTtmp[(j1)],NULL,&gpuPsums[k*samples],samples,NULL);
#ifdef DETAIL_TIMING
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gputime,start,stop);

            gpumultime+=gputime;
#endif
            //cout<<"multtime"<<gputime<<endl;
            gpu_fft32_onlyAdd(&gpuPsums[k*samples],NULL,&gpuVars1[k*samples],samples,NULL);

         }
         else if(samples <= 512*1024)
         {
            gpu_fft64_onlyMult(gpuCtxtsFFTtmp[(j1)],NULL,&gpuPsums[k*samples],samples,NULL);

            gpu_fft64_onlyAdd(&gpuPsums[k*samples],NULL,&gpuVars1[k*samples],samples,NULL);


         }
            else
         {
            gpu_fft8_onlyMult(gpuCtxtsFFTtmp[(j1)],NULL,&gpuPsums[k*samples],samples,NULL);

            gpu_fft8_onlyAdd(&gpuPsums[k*samples],NULL,&gpuVars1[k*samples],samples,NULL);


         }
         //AddMod(vars[k],vars[k],psums[k], det);
         // gpuCudaAddMod(&gpuVars[k*words], &gpuPsums[k*words], gpuN, dLength, &gpuVars[k*words], gpuCarry, gpuC_2,gpuCarry_2);

      }

      //cudaMemcpy(c_int,  &gpuVars[0*words], sizeof(uint32_t)*words, cudaMemcpyDeviceToHost);
      //temp=ZZFromBytes((const unsigned char *)c_int, sizeof(uint32_t)*words);

      if(samples > 512*1024)
      {
         cudaFree(gpuCtxtsFFTtmp[j1]);
      }

      if (j >= prms.S) break;
   }
   // Sanity-check: j should be at least S, else we've missed some terms
   //if (j < prms.S) Error("FHEkeys::processBlock: loop finished with j<S");

   //tptime +=GetTime();

   //cout<<"cycle time="<<tptime<<endl;



#ifdef DETAIL_TIMING
   cout<<"cycle gpu time="<<gputimeall<<endl;
   cudaEventDestroy(start); 
   cudaEventDestroy(stop);
   tptime =-GetTime();


   cout<<"gpu cycle time all="<<gputimeall<<endl;
   cout<<"gpu mul time all="<<gpumultime<<endl;
   cout<<"num of adds="<<numadds<<endl;
#endif
   //cout<<"num of muls="<<nummuls<<endl;
   for (k=0; k<prms.p+1; k++) {

      if(samples <= 128*1024)
      {
         gpu_fft32_onlyIFFT(&gpuVars1[k*samples],NULL,gpuE,samples,NULL);
      }
      else if(samples <= 512*1024)
         {
            gpu_fft64_onlyIFFT(&gpuVars1[k*samples],NULL,gpuE,samples,NULL);
      }
      else
      {
         gpu_fft8_onlyIFFT(&gpuVars1[k*samples],NULL,gpuE,samples,NULL);
      }

      if (samples <= 512*1024){
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
   }
      else 
      {
         gpu_istep1(gpuE, gpuF, samples, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      }

      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);

      if(samples <= 128*1024){

         //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
         gpu_set_block_count(64);
         gpu_step1_ld3(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples, NULL);
         gpu_istep0_st3(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
         gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

         //  gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit
         gpuStp = gpuTU+q;
         //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit

         //cudaEventRecord(start,0); 

         gpu_set_block_count(64);
         gpu_step1_ld3(gpuStp, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples, NULL);
         gpu_istep0_st3(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
         gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);

      }

      else if(samples <= 512*1024)
      {
         //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
         gpu_set_block_count(64);
         gpu_step1_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples*2, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples*2, NULL);
         gpu_fft128(gpuC, gpuD, gpuE, samples*2, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples*2, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples*2, NULL);
         gpuOut=gpuE;
         gpu_carry_by2((uint32_t *)gpuOut, samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
         gpu_resolve_by2(samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

         //  gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit
         gpuStp = gpuTU+q;
         //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit

         //cudaEventRecord(start,0); 


         gpu_set_block_count(64);
         gpu_step1_ld2(gpuStp, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_fft64(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
         gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
      }
      else
      {

         uint32_t* cput = gpuPull32(gpuT,tLength);
         uint32_t* cpuu = gpuPull32(gpuU,uLength);
         ZZ zt = ZZFromBytes((unsigned char*)cput,tLength*4);
         ZZ ut = ZZFromBytes((unsigned char*)cpuu,uLength*4);
         zt *= ut;
         zt>>=(q*32);
         BytesFromZZ((unsigned char*)cpuu,zt,uLength*4);
         cudaMemcpy(gpuTU,(unsigned char*)cpuu,uLength*4,cudaMemcpyHostToDevice);
         free(cput);
         free(cpuu);

         //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
         /*gpu_set_block_count(64);
         gpu_step1_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples*2, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples*2, NULL);
         gpu_fft128(gpuC, gpuD, gpuE, samples*2, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples*2, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples*2, NULL);
         gpuOut=gpuE;
         gpu_carry_by2((uint32_t *)gpuOut, samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
         gpu_resolve_by2(samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);*/

         //  gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit
         gpuStp = gpuTU+q;
         //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit

         //cudaEventRecord(start,0); 


         gpu_set_block_count(64);
         gpu_step2_ld2(gpuTU, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
         gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_step0_fft8_istep(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1(gpuE, gpuF, samples, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
         gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
      }

      gpuCudaSub (gpuT, gpuZ, gpuR, xLength, gpuCarry);         

      cudaMemset(&gpuVars[k*words], 0, sizeof(uint32_t)*words);
      gpuCudaCmpSub (gpuR, gpuN, &gpuVars[k*words], xLength, gpuCarry, dLength);          //


   }


   free(c_int);

   $GPU(cudaFree(gpuCarry));
   $GPU(cudaFree(gpuCarry_2));

   $GPU(cudaFree(gpuC_2));
   $GPU(cudaFree(gpuPsums));
   $GPU(cudaFree(gpuVars1));


   $GPU(cudaFree(gpuT));
   $GPU(cudaFree(gpuTU));
   $GPU(cudaFree(gpuS));
   $GPU(cudaFree(gpuZ));
   $GPU(cudaFree(gpuR));
   $GPU(cudaFree(gpuA));
   $GPU(cudaFree(gpuB));
   $GPU(cudaFree(gpuC));
   $GPU(cudaFree(gpuD));
   $GPU(cudaFree(gpuE));
   $GPU(cudaFree(gpuF));

   if(samples  <= 128*1024)
   {


   }
   else if(samples <= 512*1024){

      for(int j = 0; j < nCtxts; j ++)
      {
         $GPU(cudaFree(gpuCtxtsFFTtmp[j]));

      }

   }
}

//GPU code for Grade-school addition. It thas the same precedure with the Gentry's code
// Use the grade-school algorithm to add up these s (p+1)-bit numbers.
void FHEkeys::gpuGradeSchoolAdd(uint32_t *gpuCtxt, uint32_t *gpuVars, uint32_t *gpuSp)const
{
   long i,j;

   int NumCols=5, NumRows=15, spLength=9;

   long s = 15;

   uint32_t    *gpuTmp;
   gpuTmp = gpuAlloc32(words);

   uint32_t   *gpuCarry;
   uint32_t   *gpuCarry_2;
   uint32_t   *gpuC_2;
   gpuCarry = gpuAlloc32(2*words);
   gpuCarry_2 = gpuAlloc32(2*words);
   gpuC_2 = gpuAlloc32(words);

   uint32_t    *gpuT, *gpuTU, *gpuS, *gpuZ;         //Barret Multiplication
   uint32_t    *gpuR;
   uint2       *gpuA, *gpuB, *gpuC, *gpuD, *gpuE, *gpuF, *gpuOut;
   uint32_t    tLength, nLength, xLength, uLength, tuLength, sLength, zLength;
   tLength = 2*words;
   uLength = words;
   nLength = words;
   xLength = words;
   tuLength = tLength + uLength;
   sLength = tuLength - q;
   zLength = sLength + nLength;

   gpuT=gpuAlloc32(tLength);          //Barret Multiplication
   gpuTU=gpuAlloc32(tuLength);
   gpuS=gpuAlloc32(sLength);
   gpuZ=gpuAlloc32(zLength);
   gpuR=gpuAlloc32(zLength);
   gpuA=gpuAlloc64(samples*2);
   gpuB=gpuAlloc64(samples*2);
   gpuC=gpuAlloc64(samples*2);
   gpuD=gpuAlloc64(samples*2);
   gpuE=gpuAlloc64(samples*2);
   gpuF=gpuAlloc64(samples*2);

   // add columns from right to left, upto column -1
   for (j=NumCols-1; j>0; j--) { 

      long s2=15;    

      long log = NextPowerOfTwo(s); // (log of) # of carry bits to compute
      if (log > j) log = j;     // no more carry than what can reach col 0
      if ((1L<<log) > s) log--; // no more carry than what s bits can produce

      int deg = 1L<<log;

      //set(sp[0]);
      //for (i=1; i<spLength; i++) clear(sp[i]);
      cudaMemset(gpuSp, 0, sizeof(uint32_t)*words*spLength);
      cudaMemset(&gpuSp[0], 1, 1);

      //evalSymPolys1(sp, vecRows, 1L<<log, M, s); // evaluate symmetric polys
      long m, n;
      int  s1=s-1;

      ZZ tmp;
      for (m=1; m<=s; m++) {  // process the next variable, i=1,2,...  s
         for (n=min(m,deg); n>0; n--) { // compute the j'th elem. sym. poly
            //printf("m=%d n=%d\n", m, n);
            //MulMod(tmp, sp[n-1], vars[s1][j], M);
            //gpuMulMod(gpuTmp, &gpuSp[(n-1)*words], &gpuVars[(s1+j*NumCols)*words], gpuDet);
            //gpuMultiply(gpuX, xLength, gpuY, yLength, gpuT, samples);

            if(samples <= 128*1024){

               gpu_set_block_count(64);
               gpu_step1_ld3(&gpuSp[(n-1)*words], xLength, gpuA, &gpuVars[(j+s1*NumCols)*words], xLength, gpuB, samples, NULL);
               gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
               gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
               gpu_istep1_twid(gpuE, gpuF, samples, NULL);
               gpu_istep0_st3(gpuF, gpuE, samples, NULL);
               gpuOut=gpuE;
               gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
               gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);

               //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
               gpu_set_block_count(64);
               gpu_step1_ld3(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples, NULL);
               gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
               gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
               gpu_istep1_twid(gpuE, gpuF, samples, NULL);
               gpu_istep0_st3(gpuF, gpuE, samples, NULL);
               gpuOut=gpuE;
               gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
               gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

               //       gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit

               //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
               gpu_set_block_count(64);
               gpu_step1_ld3(gpuTU+q, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
               gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
               gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
               gpu_istep1_twid(gpuE, gpuF, samples, NULL);
               gpu_istep0_st3(gpuF, gpuE, samples, NULL);
               gpuOut=gpuE;
               gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
               gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
            }

            else if(samples <= 512*1024)
            {
               gpu_set_block_count(64);
               gpu_step1_ld2(&gpuSp[(n-1)*words], xLength, gpuA, &gpuVars[(j+s1*NumCols)*words], xLength, gpuB, samples, NULL);
               gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
               gpu_fft64(gpuC, gpuD, gpuE, samples, NULL);
               gpu_istep1_twid(gpuE, gpuF, samples, NULL);
               gpu_istep0_st2(gpuF, gpuE, samples, NULL);
               gpuOut=gpuE;
               gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
               gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);

               //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
               gpu_set_block_count(64);
               gpu_step1_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples*2, NULL);
               gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples*2, NULL);
               gpu_fft128(gpuC, gpuD, gpuE, samples*2, NULL);
               gpu_istep1_twid(gpuE, gpuF, samples*2, NULL);
               gpu_istep0_st2(gpuF, gpuE, samples*2, NULL);
               gpuOut=gpuE;
               gpu_carry_by2((uint32_t *)gpuOut, samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
               gpu_resolve_by2(samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

               //       gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit

               //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
               gpu_set_block_count(64);
               gpu_step1_ld2(gpuTU+q, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
               gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
               gpu_fft64(gpuC, gpuD, gpuE, samples, NULL);
               gpu_istep1_twid(gpuE, gpuF, samples, NULL);
               gpu_istep0_st2(gpuF, gpuE, samples, NULL);
               gpuOut=gpuE;
               gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
               gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);


            }

            else
            {
               gpu_set_block_count(64);
               gpu_step2_ld2(&gpuSp[(n-1)*words], xLength, gpuA, &gpuVars[(j+s1*NumCols)*words], xLength, gpuB, samples, NULL);
               gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
               gpu_step0_fft8_istep(gpuC, gpuD, gpuE, samples, NULL);
               gpu_istep1(gpuE, gpuF, samples, NULL);
               gpu_istep0_st2(gpuF, gpuE, samples, NULL);
               gpuOut=gpuE;
               gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
               gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);


               uint32_t* cput = gpuPull32(gpuT,tLength);
               uint32_t* cpuu = gpuPull32(gpuU,uLength);
               ZZ zt = ZZFromBytes((unsigned char*)cput,tLength*4);
               ZZ ut = ZZFromBytes((unsigned char*)cpuu,uLength*4);
               zt *= ut;
               zt>>=(q*32);
               BytesFromZZ((unsigned char*)cpuu,zt,uLength*4);
               cudaMemcpy(gpuTU,(unsigned char*)cpuu,uLength*4,cudaMemcpyHostToDevice);
               free(cput);
               free(cpuu);

               //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
               /*gpu_set_block_count(64);
               gpu_step1_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples*2, NULL);
               gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples*2, NULL);
               gpu_fft128(gpuC, gpuD, gpuE, samples*2, NULL);
               gpu_istep1_twid(gpuE, gpuF, samples*2, NULL);
               gpu_istep0_st2(gpuF, gpuE, samples*2, NULL);
               gpuOut=gpuE;
               gpu_carry_by2((uint32_t *)gpuOut, samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
               gpu_resolve_by2(samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);*/

               //       gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit

               //gpuMultiply(gpuS, xLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
               gpu_set_block_count(64);
               gpu_step2_ld2(gpuTU, xLength, gpuA, gpuN, nLength, gpuB, samples, NULL);
               gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
               gpu_step0_fft8_istep(gpuC, gpuD, gpuE, samples, NULL);
               gpu_istep1(gpuE, gpuF, samples, NULL);
               gpu_istep0_st2(gpuF, gpuE, samples, NULL);
               gpuOut=gpuE;
               gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);
               gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, xLength, NULL);

            }


            gpuCudaSub (gpuT, gpuZ, gpuR, tLength, gpuCarry);          //R < -- T-Z  2*n-bit   
            gpuCudaCmpSub (gpuR, gpuN, gpuTmp, xLength, gpuCarry, dLength);          //         

            //AddMod(sp[n], sp[n], tmp, M); // out[j] += out[j-1] * vars.top() mod M
            //gpuAddMod(&sp[n*words], &sp[n*words], gpuTmp, gpuDet);
            gpuCudaAddMod(&gpuSp[n*words], gpuTmp, gpuN, dLength, &gpuSp[n*words], gpuCarry, gpuC_2,gpuCarry_2);
         }
         s1--;
      }

      // The carry bits from this column are sp[2],sp[4],sp[8]... The result
      // for that column is in sp[1] (but for most columns we don't need it)

      long k = 2;
      for (long j2=j-1; j2>=0 && k< deg+1; j2--) {
         //vars[s2][j2] = sp[k]; gpuVars[(s2+j2*NumCols)*words] = gpuSp[k*words]
         cudaMemcpy(&gpuVars[(j2+s2*NumCols)*words], &gpuSp[k*words], sizeof(uint32_t)*words, cudaMemcpyDeviceToDevice);
         s2++;
         k <<= 1;
      }

      s++;
   }

   // The result from column -1 is in sp[1], add to it all the bit in column 0
   //c = vars[0][0]; //gpuCtxt
   cudaMemcpy(gpuCtxt, &gpuVars[0], sizeof(uint32_t)*words, cudaMemcpyDeviceToDevice);

   for(i=1; i<19; i++) {
      //AddMod(c, c, vars[i][0], M);
      gpuCudaAddMod(gpuCtxt, &gpuVars[i*NumCols*words], gpuN, dLength, gpuCtxt, gpuCarry, gpuC_2,gpuCarry_2);
   }

   //AddMod(c, c, sp[1], M);
   gpuCudaAddMod(gpuCtxt, &gpuSp[words], gpuN, dLength, gpuCtxt, gpuCarry, gpuC_2,gpuCarry_2);

   $GPU(cudaFree(gpuTmp));
   $GPU(cudaFree(gpuCarry));
   $GPU(cudaFree(gpuCarry_2));
   $GPU(cudaFree(gpuC_2));
   $GPU(cudaFree(gpuT));
   $GPU(cudaFree(gpuTU));
   $GPU(cudaFree(gpuS));
   $GPU(cudaFree(gpuZ));
   $GPU(cudaFree(gpuR));
   $GPU(cudaFree(gpuA));
   $GPU(cudaFree(gpuB));
   $GPU(cudaFree(gpuC));
   $GPU(cudaFree(gpuD));
   $GPU(cudaFree(gpuE));
   $GPU(cudaFree(gpuF));

}

//#include "my_grade.h"
// Perform homomorphic decryption of the cipehrtext c

// Precompute the FFT transforms of the secret keys and prestore them 
// in GPU. You can refer to our paper to understand this part.
void FHEkeys::gpuRecryptPre(uint2 **gpuCtxtsFFT,uint32_t *gpuCtxts)
{
   uint2** prextp;


   prextp = new uint2*[prms.s];

   prex = (void**)prextp;
   for(int i = 0; i < prms.s;i++){
      ZZ factor = pkBlocks[i].x;


      uint2* gpuA,*gpuC;
      uint2* gpuB,*gpuD;
      gpuA = gpuAlloc64(samples);
      gpuC = gpuAlloc64(samples);
      gpuB = gpuAlloc64(samples);
      gpuD = gpuAlloc64(samples);
      prextp[i] = gpuAlloc64(samples);

      uint32_t   *tempfactor1;

      tempfactor1 = gpuAlloc32(words);

      uint32_t   *cputempfactor1;

      cputempfactor1 = (uint32_t *)malloc(sizeof(uint32_t)*words);

      BytesFromZZ((unsigned char *)cputempfactor1, factor, sizeof(uint32_t)*words);

      cudaMemcpy(tempfactor1,  cputempfactor1, sizeof(uint32_t)*words, cudaMemcpyHostToDevice);

      gpu_set_block_count(64);

      if(samples <= 128*1024){
         gpu_step1_ld3_onlyX(tempfactor1, words, gpuA, samples, NULL);
         gpu_step0_twid_onlyX(gpuA, gpuC,  samples, NULL);

         gpu_fft32_onlyX(gpuC,prextp[i],samples, NULL);
      }
      else if(samples <= 512*1024)
      {
         gpu_step1_ld2(tempfactor1, words, gpuA, tempfactor1, words, gpuC, samples, NULL);
         gpu_step0_twid_onlyX(gpuA, gpuC,  samples, NULL);

         gpu_fft64_onlyX(gpuC,prextp[i],samples, NULL);

      }
      else
      {
         gpu_step2_ld2(tempfactor1, words, gpuA, tempfactor1, words, gpuB, samples, NULL);
         gpu_step1(gpuA, gpuC, gpuB, gpuD,  samples, NULL);

         gpu_fft8_onlyX(gpuC,prextp[i],samples, NULL);
      }

      free(cputempfactor1);
      $GPU(cudaFree(gpuA));
      $GPU(cudaFree(gpuC));
      $GPU(cudaFree(gpuB));
      $GPU(cudaFree(gpuD));
      $GPU(cudaFree(tempfactor1));

   }

   uint2* gpuA,*gpuC,*gpuB,*gpuD;

   gpuA = gpuAlloc64(samples);
   gpuC = gpuAlloc64(samples);
   gpuB = gpuAlloc64(samples);
   gpuD = gpuAlloc64(samples);
   uint2* preFFT;
   long nCtxts = mChoose2(prms.S);

   uint2* gpuCtxtsFFTtmp;

   gpuCtxtsFFTtmp = gpuAlloc64(samples);

   for(int i = 0; i < nCtxts*prms.s; i ++)
   {

      uint32_t* gpuCtxt2;

      gpuCtxt2 = gpuPush32(&gpuCtxts[i*words],samples);

      if(samples <= 512*1024)
      {
      gpu_set_block_count(64);
      gpu_step1_ld2(gpuCtxt2 , words, gpuA, gpuCtxt2 , words, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      }
      else
      {
         gpu_set_block_count(64);
      gpu_step2_ld2(gpuCtxt2 , words, gpuA, gpuCtxt2 , words, gpuB, samples, NULL);
      gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      }

      if(samples <= 128*1024)
      {
         gpu_fft32_onlyX(gpuC,gpuCtxtsFFTtmp ,samples, NULL);
      }
      else if(samples <= 512*1024)
         {
            gpu_fft64_onlyX(gpuC,gpuCtxtsFFTtmp ,samples, NULL);
      }
      else
      {
         gpu_fft8_onlyX(gpuC,gpuCtxtsFFTtmp ,samples, NULL);
      }
      if(samples <= 128*1024)
      {
         $GPU(cudaMalloc((void **)&gpuCtxtsFFT[i], sizeof(uint2)*samples));

         $GPU(cudaMemcpy(gpuCtxtsFFT[i], gpuCtxtsFFTtmp, sizeof(uint2)*samples, cudaMemcpyDeviceToDevice));
      }
      else
      {
         gpuCtxtsFFT[i]=(uint2 *)malloc(sizeof(uint2)*samples);
         $GPU(cudaMemcpy(gpuCtxtsFFT[i], gpuCtxtsFFTtmp, sizeof(uint2)*samples, cudaMemcpyDeviceToHost));
      }

      $GPU(cudaFree(gpuCtxt2));

   }

   $GPU(cudaFree(gpuA));
   $GPU(cudaFree(gpuC));
   $GPU(cudaFree(gpuB));
   $GPU(cudaFree(gpuD));

   $GPU(cudaFree(gpuCtxtsFFTtmp));
}

void FHEkeys::gpuRecrypt(FHEctxt& c) const
{
   // From each public-key block we compute an encrypted (p+1)-bit vector
   //mat_ZZ vars(INIT_SIZE, prms.s, prms.p +1);

   ZZ     temp;
   uint32_t *gpuDet;
   gpuDet = gpuN;


   uint32_t    spLength = 9;

   uint32_t    *gpuCtxt;
   uint32_t    *gpuVars;
   uint32_t    *gpuSp;
   uint32_t    *c_int;

   c_int=(uint32_t *)malloc(sizeof(uint32_t)*words);
   gpuCtxt = gpuAlloc32(words);  
   gpuVars = gpuAlloc32(words*(prms.s+4)*(prms.p+1));
   gpuSp = gpuAlloc32(words*spLength);

   for(long i=0; i<prms.s; i++) FHEkeys::gpuProcessBlock(&gpuVars[i*(prms.p+1)*words], gpuCtxtsFFT, c, i, temp);

   gpuGradeSchoolAdd(gpuCtxt, gpuVars,gpuSp);

   cudaMemcpy(c_int, gpuCtxt, sizeof(uint32_t)*words, cudaMemcpyDeviceToHost);

   c=ZZFromBytes((const unsigned char *)c_int, sizeof(uint32_t)*words);

   free(c_int);
   $GPU(cudaFree(gpuCtxt));  
   $GPU(cudaFree(gpuVars));
   $GPU(cudaFree(gpuSp));

}

//Precompute the FFT transforms of the powers of root
void encrypt_FHE_pre(uint2** gpuPwr,uint2** gpuPwr64, int samples, uint32_t *gpuU, uint32_t uLength, uint32_t *gpuN, uint32_t nLength, uint32_t q, uint32_t *power, uint32_t pwrLength, uint32_t *pwrFac, uint32_t *pwrFixFac,int windowsize, int windownum){
   
   uint32_t    *gpuRand, *gpuPwrFac, *gpuPwrFixFac;
   uint32_t    *gpuVal ;
   uint32_t    *gpuSum, *gpuPwrtp, *gpuCarry, *gpuCarry2,*gpuC_2,  *gpuSum1;
   uint32_t    *gpuT, *gpuTU, *gpuS, *gpuZ;         //Barret Multiplication
   uint32_t    *gpuR;
   uint2       *gpuA, *gpuB, *gpuC, *gpuD, *gpuE, *gpuF, *gpuOut;
   uint32_t    tLength, tuLength, sLength, zLength;
   tLength = 2*uLength;
   tuLength = tLength + uLength;
   sLength = tuLength - q;
   zLength = sLength + uLength;

   gpuPwrtp= gpuPush32(power, pwrLength);
   gpuPwrFac= gpuPush32(pwrFac, uLength);
   gpuPwrFixFac= gpuPush32(pwrFixFac, uLength);

   gpuVal = gpuAlloc32(uLength);
   gpuCarry = gpuAlloc32(tLength);

   gpuC_2 = gpuAlloc32(uLength);
   cout<<"samples = "<<samples;
   cout<<"ulength = "<<uLength;
   cout<<"tlength = "<<tLength;
   if(windownum < 64)
   {
      *gpuPwr = gpuAlloc64(samples*windowsize*2);
      *gpuPwr64 = gpuAlloc64(samples*windownum);
   }
   else
   {
      *gpuPwr = (uint2*)malloc(((uint64_t)samples)*windowsize*16);   
      *gpuPwr64 = (uint2*)malloc(((uint64_t)samples)*windownum*8);
      cout<<"memory!"<<endl;
   }
   gpuT=gpuAlloc32(tLength);          //Barret Multiplication
   cout<<tuLength<<endl;
   gpuTU=gpuAlloc32(tuLength);
   cout<<q<<endl;
   cout<<sLength<<endl;
   gpuS=gpuAlloc32(sLength);
   gpuZ=gpuAlloc32(zLength);
   gpuR=gpuAlloc32(zLength);
   
   gpuA=gpuAlloc64(samples*2);
   gpuB=gpuAlloc64(samples*2);
   gpuC=gpuAlloc64(samples*2);
   gpuD=gpuAlloc64(samples*2);
   gpuE=gpuAlloc64(samples*2);
   gpuF=gpuAlloc64(samples*2);
   
   uint2 *gpuPwr2;
   uint2 *gpuPwr642;
   gpuPwr2 = *gpuPwr;
   if(windownum < 64)gpuPwr642= *gpuPwr64;
   else 
   {
   }
   for(int i = 0; i < windowsize; i ++)
   {      
      //cout<<"window "<<i<<endl;
      uint2* tppwr;
      uint2* tppwr2;
      if (samples <= 512*1024)
      {
         gpuCudaSub (gpuN, &gpuPwrtp[i*uLength], gpuT, uLength, gpuCarry);
         gpu_set_block_count(64);
         gpu_step1_ld2(&gpuPwrtp[i*uLength], uLength, gpuA, gpuT, uLength, gpuB, samples, NULL);
         
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         if(windownum < 64)
         {
            tppwr = &gpuPwr2[i*samples];
            tppwr2 = &gpuPwr2[(i+64)*samples];

         }
         else 
         {
            tppwr = gpuAlloc64(samples);
            tppwr2 = gpuAlloc64(samples);

         }
         if(samples <= 128*1024){
            gpu_fft32_onlyX(gpuC,tppwr,samples, NULL);
            gpu_fft32_onlyX(gpuD,tppwr2,samples, NULL);
         }
         else
         {
            gpu_fft64_onlyX(gpuC,tppwr,samples, NULL);
            gpu_fft64_onlyX(gpuD,tppwr2,samples, NULL);
         }

      }
      else
      {
         gpuCudaSub (gpuN, &gpuPwrtp[i*uLength], gpuT, uLength, gpuCarry);
         gpu_set_block_count(64);
         gpu_step2_ld2(&gpuPwrtp[i*uLength], uLength, gpuA, gpuT, uLength, gpuB, samples, NULL);
         gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
                  
         if(windownum < 64)
         {
            tppwr = &gpuPwr2[i*samples];
            tppwr2 = &gpuPwr2[(i+64)*samples];

         }
         else {
            tppwr = gpuAlloc64(samples);
            tppwr2 = gpuAlloc64(samples);

         }
         
         gpu_fft8_onlyX(gpuC,tppwr,samples, NULL);
         gpu_fft8_onlyX(gpuD,tppwr2,samples, NULL);
         
      }


      if(windownum < 64)
      {
      }
      else
      {
         cudaMemcpy(&gpuPwr2[i*samples],tppwr,8*samples,cudaMemcpyDeviceToHost);
         cudaMemcpy(&gpuPwr2[(i+64)*samples],tppwr2,8*samples,cudaMemcpyDeviceToHost);
         cudaFree(tppwr);
         cudaFree(tppwr2);
      }
   }


   for(int i = 0; i < windownum; i ++)
   {
      gpu_set_block_count(64);
   
      if (windownum < 64){
         gpu_step1_ld2(gpuPwrFac, uLength, gpuA, gpuPwrFac, uLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         if(samples <= 128*1024)
         {
            gpu_fft32_onlyX(gpuC,&gpuPwr642[i*(samples)],samples, NULL);
         }
         else gpu_fft64_onlyX(gpuC,&gpuPwr642[i*(samples)],samples, NULL);
      }
      else
      {
         gpuPwr642 = gpuAlloc64(samples);
         if(samples <= 128*1024)
         {
            gpu_step1_ld2(gpuPwrFac, uLength, gpuA, gpuPwrFac, uLength, gpuB, samples, NULL);
            gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
            gpu_fft32_onlyX(gpuC,gpuPwr642,samples, NULL);
         }
         else if(samples <= 512*1024)
         {
            gpu_step1_ld2(gpuPwrFac, uLength, gpuA, gpuPwrFac, uLength, gpuB, samples, NULL);
            gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
            gpu_fft64_onlyX(gpuC,gpuPwr642,samples, NULL);
         }
         else
         {
            gpu_step2_ld2(gpuPwrFac, uLength, gpuA, gpuPwrFac, uLength, gpuB, samples, NULL);
            gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
            gpu_fft8_onlyX(gpuC,gpuPwr642,samples, NULL);
         }
         uint2* tp = *gpuPwr64;
         cudaMemcpy(tp+i*samples, gpuPwr642, 8*samples, cudaMemcpyDeviceToHost);
         cudaFree(gpuPwr642);
      }

      if(i ==windownum)break;

      if(samples <= 128*1024)
      {

         gpu_set_block_count(64);
         gpu_step1_ld3(gpuPwrFac, uLength, gpuA, gpuPwrFixFac, uLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples, NULL);
         gpu_istep0_st3(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
         gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);

         //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
         gpu_set_block_count(64);
         gpu_step1_ld3(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples, NULL);
         gpu_istep0_st3(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
         gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

         //gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit

         //gpuMultiply(gpuS, nLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
         gpu_set_block_count(64);
         gpu_step1_ld3(gpuTU+q, uLength, gpuA, gpuN, uLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples, NULL);
         gpu_istep0_st3(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, tLength, NULL);
         gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuZ, tLength, NULL);

         gpuCudaSub (gpuT, gpuZ, gpuR, tLength, gpuCarry);          //R < -- T-Z  2*n-bit   
         gpuCudaCmpSub (gpuR, gpuN, gpuPwrFac, uLength, gpuCarry,nLength);          //
      }
      else if (samples <= 512*1024)
      {
         gpu_set_block_count(64);
         gpu_step1_ld2(gpuPwrFac, uLength, gpuA, gpuPwrFixFac, uLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_fft64(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
         gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
         //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
         gpu_set_block_count(64);
         gpu_step1_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples*2, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples*2, NULL);
         gpu_fft128(gpuC, gpuD, gpuE, samples*2, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples*2, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples*2, NULL);
         gpuOut=gpuE;
         gpu_carry_by2((uint32_t *)gpuOut, samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
         gpu_resolve_by2(samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

         //gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit
         //gpuMultiply(gpuS, nLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
         gpu_set_block_count(64);
         gpu_step1_ld2(gpuTU+q, uLength, gpuA, gpuN, uLength, gpuB, samples, NULL);
         gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_fft64(gpuC, gpuD, gpuE, samples, NULL);
         gpu_istep1_twid(gpuE, gpuF, samples, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);
         gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);

         gpuCudaSub (gpuT, gpuZ, gpuR, tLength, gpuCarry);          //R < -- T-Z  2*n-bit   
         gpuCudaCmpSub (gpuR, gpuN, gpuPwrFac, uLength, gpuCarry,nLength);          //

      }
      else 
      {

         gpu_set_block_count(64);
         gpu_step2_ld2(gpuPwrFac, uLength, gpuA, gpuPwrFixFac, uLength, gpuB, samples, NULL);
         gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         
         gpu_step0_fft8_istep(gpuC, gpuD, gpuE, samples, NULL);
         
         gpu_istep1(gpuE, gpuF, samples, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         
         gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
         gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);

         uint32_t* cput = gpuPull32(gpuT,tLength);
         uint32_t* cpuu = gpuPull32(gpuU,uLength);
         ZZ zt = ZZFromBytes((unsigned char*)cput,tLength*4);
         ZZ ut = ZZFromBytes((unsigned char*)cpuu,uLength*4);
         zt *= ut;
         zt>>=(q*32);
         BytesFromZZ((unsigned char*)cpuu,zt,uLength*4);
         cudaMemcpy(gpuTU,(unsigned char*)cpuu,uLength*4,cudaMemcpyHostToDevice);
         free(cput);
         free(cpuu);
         //gpuMultiply(gpuT, tLength, gpuU, uLength, gpuTU, samples); //TU <-- T*U  2*n-bit
         /*gpu_set_block_count(64);
         gpu_step2_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples, NULL);
         gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_step0_twid(gpuC, gpuA, gpuD, gpuB, samples, NULL);
         gpu_fft16(gpuA, gpuB, gpuE, samples, NULL);
         gpu_istep2_twid(gpuE, gpuF, samples, NULL);
         gpu_istep1(gpuF, gpuE, samples, NULL);
         gpu_istep0_st2(gpuE, gpuF, samples, NULL);
         gpuOut=gpuF;
         gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
         gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);*/

         //gpuCudaDivQ(gpuS, gpuTU, tuLength, q);                     //S <-- TU/Q  3*n-bit - 2*n bit

         //gpuMultiply(gpuS, nLength, gpuN, nLength, gpuZ, samples);  //Z <-- S*N   n-bit
         gpu_set_block_count(64);
         gpu_step2_ld2(gpuTU, uLength, gpuA, gpuN, uLength, gpuB, samples, NULL);
         gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
         gpu_step0_fft8_istep(gpuC, gpuD, gpuE, samples, NULL);         
         gpu_istep1(gpuE, gpuF, samples, NULL);
         gpu_istep0_st2(gpuF, gpuE, samples, NULL);
         gpuOut=gpuE;
         gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);
         gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);

         gpuCudaSub (gpuT, gpuZ, gpuR, tLength, gpuCarry);          //R < -- T-Z  2*n-bit   
         gpuCudaCmpSub (gpuR, gpuN, gpuPwrFac, uLength, gpuCarry,nLength);          //

      }

   }

   $GPU(cudaFree(gpuPwrtp));
   $GPU(cudaFree(gpuPwrFac)); 
   $GPU(cudaFree(gpuPwrFixFac));
   $GPU(cudaFree(gpuVal));    
   $GPU(cudaFree(gpuCarry));
   $GPU(cudaFree(gpuC_2));

   $GPU(cudaFree(gpuT));
   $GPU(cudaFree(gpuTU));
   $GPU(cudaFree(gpuS));
   $GPU(cudaFree(gpuZ));
   $GPU(cudaFree(gpuR));
   $GPU(cudaFree(gpuA));
   $GPU(cudaFree(gpuB));
   $GPU(cudaFree(gpuC));
   $GPU(cudaFree(gpuD));
   $GPU(cudaFree(gpuE));
   $GPU(cudaFree(gpuF));



}

void verifykey(uint2 *target,uint2 *base, uint32_t uLength,uint32_t *gpuN,uint32_t nLength,int power, int samples)
{
   uint2 *gpuA=gpuAlloc64(samples*2);
   uint2 *gpuB=gpuAlloc64(samples*2);
   uint2 *gpuC=gpuAlloc64(samples*2);
   uint2 *gpuD=gpuAlloc64(samples*2);
   uint2 *gpuE=gpuAlloc64(samples*2);
   uint2 *gpuF=gpuAlloc64(samples*2);

   gpu_fft8_onlyIFFT(target,NULL,gpuE,samples,NULL);
   
      gpu_istep1(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpu_carry_by2((uint32_t *)gpuE, samples, (uint32_t *)gpuA, (uint32_t *)gpuF, uLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, (uint32_t *)gpuB, 2*uLength, NULL);


   uint32_t* cputarget = gpuPull32((uint32_t *)gpuB,2*uLength);
   uint32_t* cpubase = gpuPull32((uint32_t *)base,uLength);
   uint32_t* cpudet = gpuPull32((uint32_t *)gpuN,nLength);
   ZZ targetZZ =  ZZFromBytes((unsigned char*)cputarget,uLength*8);
   ZZ baseZZ = ZZFromBytes((unsigned char*)cpubase,uLength*4);
   ZZ det = ZZFromBytes((unsigned char*)cpudet,nLength*4);

   ZZ tp = PowerMod(baseZZ,power,det);
   if (targetZZ%det != tp)
   {
      cout<<"wrong!"<<endl;
   }

   free(cputarget);
   free(cpubase);
   free(cpudet);
   $GPU(cudaFree(gpuA));
   $GPU(cudaFree(gpuB));
   $GPU(cudaFree(gpuC));
   $GPU(cudaFree(gpuD));
   $GPU(cudaFree(gpuE));
   $GPU(cudaFree(gpuF));   
}

//GPU codes for the encryption. It uses sliding window method in the paper
// the calculation is performed in FFT domain
void encrypt_FHE_FFT(uint32_t *z, int samples, uint32_t *gpuU, uint32_t uLength, uint32_t *gpuN, uint32_t nLength, uint32_t *s, uint32_t q, uint2 *gpuPwr, uint2 *gpuPwr64,int windowsize,int windownum){
   
   uint32_t    *gpuPwrFac, *gpuPwrFixFac;
   uint2    *gpuVal ;
   uint32_t      *gpuCarry, *gpuCarry2,*gpuC_2, *gpuSum1;
   uint2 *gpuSum;

   uint32_t    *gpuT, *gpuTU, *gpuS, *gpuZ;         //Barret Multiplication
   uint32_t    *gpuR;
   uint2       *gpuA, *gpuB, *gpuC, *gpuD, *gpuE, *gpuF, *gpuOut;
   uint32_t    tLength, tuLength, sLength, zLength;
   tLength = 2*uLength;
   tuLength = tLength + uLength;
   sLength = tuLength - q;
   zLength = sLength + uLength;


   if(windownum <64)
   {   gpuSum = gpuAlloc64(samples*windownum);
   }
   else
   {
      gpuSum = (uint2*)malloc(((uint64_t)samples)*windownum*8);

   }
   gpuVal = gpuAlloc64(samples);
   gpuCarry = gpuAlloc32(tLength);
   gpuC_2 = gpuAlloc32(uLength);

   gpuT=gpuAlloc32(tLength);          //Barret Multiplication
   gpuTU=gpuAlloc32(tuLength);
   gpuS=gpuAlloc32(sLength);
   gpuZ=gpuAlloc32(zLength);
   gpuR=gpuAlloc32(zLength);
   
   gpuA=gpuAlloc64(samples*2);
   gpuB=gpuAlloc64(samples*2);
   gpuC=gpuAlloc64(samples*2);
   gpuD=gpuAlloc64(samples*2);
   gpuE=gpuAlloc64(samples*2);
   gpuF=gpuAlloc64(samples*2);
   
   gpu_set_block_count(64);
#pragma unroll    
   for(int j=0; j<windownum; j++) {   

      bool allzero = true;
      uint2* tpsum;
      if(windownum < 64)tpsum = &gpuSum[j*samples];
      else
      {
         tpsum = gpuAlloc64(samples);
         //   cudaMemcpy(tpsum,&gpuSum[j*samples],8*samples,cudaMemcpyHostToDevice);
      }
      uint2* tppwr;
      uint2* tppwr2;
      uint2* aaa;
      uint2* bbb;
      if(windownum >= 64)
      {
         aaa = gpuAlloc64(samples);
         bbb = gpuAlloc64(samples);
      }

      for(int i=0; i<windowsize; i++) {  
         if(windownum < 64)
         {
            tppwr = &gpuPwr[i*samples];
            tppwr2 = &gpuPwr[(i+windowsize)*samples];
         }
         else
         {
            if(i == 0||i == 32){
               //   cudaMemcpy(bbb, &gpuPwr[(i+windowsize)*samples],samples*8*32,cudaMemcpyHostToDevice);
            }
            tppwr = &aaa[0];
            tppwr2 = &bbb[0];
         };

         if(s[i+j*windowsize]==1) {
            allzero = false;
            if(windownum >= 64)
            {
            cudaMemcpy(tppwr, &gpuPwr[(i)*samples],samples*8,cudaMemcpyHostToDevice);
            }
            if(samples <= 128*1024){
               gpu_fft32_onlyAdd(tppwr,NULL,tpsum,samples,NULL);
            }
            else if(samples <= 512*1024){
               gpu_fft64_onlyAdd(tppwr,NULL,tpsum,samples,NULL);
               }
            else
            {
               gpu_fft8_onlyAdd(tppwr,NULL,tpsum,samples,NULL);
            }

         }
         else if(s[i+j*windowsize]==-1) {
            allzero = false;
            if(windownum >= 64)
            {
            cudaMemcpy(tppwr2, &gpuPwr[(i+windowsize)*samples],samples*8,cudaMemcpyHostToDevice);
            }
            if(samples <= 128*1024){                 
               gpu_fft32_onlyAdd(tppwr2,NULL,tpsum,samples,NULL);
            }
            else  if(samples <= 512*1024){
               gpu_fft64_onlyAdd(tppwr2,NULL,tpsum,samples,NULL);
            }
            else
            {
               gpu_fft8_onlyAdd(tppwr2,NULL,tpsum,samples,NULL);
            }
            allzero = false;
         }
      }
      if(windownum < 64)
      {}
      else
      {
         cudaFree(aaa);
         cudaFree(bbb);
      }     

      if(!allzero){

         if(windownum < 64){
            if(samples <= 128*1024)
            {
               gpu_fft32_onlyMult(&gpuPwr64[j*samples],NULL,tpsum,samples,NULL);
               gpu_fft32_onlyAdd(tpsum,NULL,gpuVal,samples,NULL);
            }
            else  if(samples <= 512*1024)
            {
               gpu_fft64_onlyMult(&gpuPwr64[j*samples],NULL,tpsum,samples,NULL);
               gpu_fft64_onlyAdd(tpsum,NULL,gpuVal,samples,NULL);
            }
            else
            {
               gpu_fft8_onlyMult(&gpuPwr64[j*samples],NULL,tpsum,samples,NULL);
               gpu_fft8_onlyAdd(tpsum,NULL,gpuVal,samples,NULL);
            }
         }
         else
         {
            uint2* tp;
            tp = gpuAlloc64(samples);
            //cout<<"dasd"<<endl;
            cudaMemcpy(tp, &gpuPwr64[j*samples], 8*samples, cudaMemcpyHostToDevice);
            if(samples <= 128*1024)
            {
               gpu_fft32_onlyMult(tp,NULL,tpsum,samples,NULL);
               gpu_fft32_onlyAdd(tpsum,NULL,gpuVal,samples,NULL);
            }
            else if(samples <= 512*1024)
            {
               gpu_fft64_onlyMult(tp,NULL,tpsum,samples,NULL);
               gpu_fft64_onlyAdd(tpsum,NULL,gpuVal,samples,NULL);
            }
            else
            {
               gpu_fft8_onlyMult(tp,NULL,tpsum,samples,NULL);
               gpu_fft8_onlyAdd(tpsum,NULL,gpuVal,samples,NULL);
            }
         cudaFree(tp);
         }
      }

      if(windownum < 64){}
      else
      {
         // cudaMemcpy( &gpuSum[j*samples], tpsum, 8*samples, cudaMemcpyDeviceToHost);
         cudaFree(tpsum);
      }
   }
   if(samples <= 128*1024){  
      gpu_fft32_onlyIFFT(gpuVal,NULL,gpuE,samples,NULL);
   }
   else if(samples <= 512*1024)
   {
      gpu_fft64_onlyIFFT(gpuVal,NULL,gpuE,samples,NULL);
   }
   else
   {
      gpu_fft8_onlyIFFT(gpuVal,NULL,gpuE,samples,NULL);
   }
   if(samples <= 512*1024)
   {
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
   }
   else
   {
      
      gpu_istep1(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
   }
   gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
   gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuT, tLength, NULL);
   
   if(samples <= 128*1024){
      gpu_set_block_count(64);

      gpu_step1_ld3(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft32(gpuC, gpuD, gpuE, samples*2, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st3(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
      gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

      gpu_set_block_count(64);
      gpu_step1_ld3(gpuTU+q, sLength, gpuA, gpuN, uLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft32(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st3(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by3((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);
      gpu_resolve_by3(samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);
   }
   else if (samples <= 512*1024)
   {
      gpu_set_block_count(64);
      gpu_step1_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples*2, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples*2, NULL);
      gpu_fft128(gpuC, gpuD, gpuE, samples*2, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples*2, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples*2, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
      gpu_resolve_by2(samples*2, (uint32_t *)gpuA, gpuTU, tuLength, NULL);

      gpu_set_block_count(64);
      gpu_step1_ld2(gpuTU+q, sLength, gpuA, gpuN, uLength, gpuB, samples, NULL);
      gpu_step0_twid(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_fft64(gpuC, gpuD, gpuE, samples, NULL);
      gpu_istep1_twid(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);
   }
   else
   {
         uint32_t* cput = gpuPull32(gpuT,tLength);
         uint32_t* cpuu = gpuPull32(gpuU,uLength);
         ZZ zt = ZZFromBytes((unsigned char*)cput,tLength*4);
         ZZ ut = ZZFromBytes((unsigned char*)cpuu,uLength*4);
         zt *= ut;
         zt>>=(q*32);
         BytesFromZZ((unsigned char*)cpuu,zt,uLength*4);
         cudaMemcpy(gpuTU,(unsigned char*)cpuu,uLength*4,cudaMemcpyHostToDevice);
         free(cput);
         free(cpuu);

      /*gpu_set_block_count(64);
      gpu_step2_ld2(gpuT, tLength, gpuA, gpuU, uLength, gpuB, samples, NULL);
      gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_step0_twid(gpuC, gpuA, gpuD, gpuB, samples, NULL);
      gpu_fft16(gpuA, gpuB, gpuE, samples, NULL);
      gpu_istep2_twid(gpuE, gpuF, samples, NULL);
      gpu_istep1(gpuF, gpuE, samples, NULL);
      gpu_istep0_st2(gpuE, gpuF, samples, NULL);
      gpuOut=gpuF;
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuTU, tuLength, NULL);*/

      gpu_set_block_count(64);
      gpu_step2_ld2(gpuTU, sLength, gpuA, gpuN, uLength, gpuB, samples, NULL);
      gpu_step1(gpuA, gpuC, gpuB, gpuD, samples, NULL);
      gpu_step0_fft8_istep(gpuC, gpuD, gpuE, samples, NULL);         
      gpu_istep1(gpuE, gpuF, samples, NULL);
      gpu_istep0_st2(gpuF, gpuE, samples, NULL);
      gpuOut=gpuE;
      gpu_carry_by2((uint32_t *)gpuOut, samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);
      gpu_resolve_by2(samples, (uint32_t *)gpuA, gpuZ, zLength, NULL);
   }

   gpuCudaSub (gpuT, gpuZ, gpuR, tLength, gpuCarry);          //R < -- T-Z  2*n-bit 
   cudaMemset(gpuT, 0, sizeof(uint32_t)*tLength);
   gpuCudaCmpSub (gpuR, gpuN, gpuT, uLength, gpuCarry, nLength);      

   $GPU(cudaMemcpy(z, gpuT, sizeof(uint32_t)*(uLength), cudaMemcpyDeviceToHost)); //gpuVal


   if(windownum<64){ $GPU(cudaFree(gpuSum))}
   else free(gpuSum);
   $GPU(cudaFree(gpuVal));    
   $GPU(cudaFree(gpuCarry));
   $GPU(cudaFree(gpuC_2));

   $GPU(cudaFree(gpuT));
   $GPU(cudaFree(gpuTU));
   $GPU(cudaFree(gpuS));
   $GPU(cudaFree(gpuZ));
   $GPU(cudaFree(gpuR));
   $GPU(cudaFree(gpuA));
   $GPU(cudaFree(gpuB));
   $GPU(cudaFree(gpuC));
   $GPU(cudaFree(gpuD));
   $GPU(cudaFree(gpuE));
   $GPU(cudaFree(gpuF));

}

//Precomputation for Barrett multiplication
bool FHEkeys::gpuPreComputation0()
{
   if(inited0)return false;

   q=dLength*2*32;
   uint32_t* u;

   u=(uint32_t *)malloc(sizeof(uint32_t)*words);

   ZZ tp;

   tp = 2;

   tp =  power(tp, q); 

   tp = tp / det;

   q/=32;
   /* 
   cout<<"q="<<q;
   cout<<"detbit="<<NumBytes(det);
   cout<<"ubit="<<NumBytes(tp);
   */
   BytesFromZZ((unsigned char*)u, tp,words*4);

   gpuU = gpuPush32(u, words);

   BytesFromZZ((unsigned char*)u, det,words*4);
   gpuN = gpuPush32(u, words);

   inited0 = true;
   free(u);
   return true;
}

bool FHEkeys::gpuPreComputation1()
{
   if(inited1)return false;
   if(!inited0)return false;
   getFreeMemory() ;

   ZZ tp;

   {
      cout<<"Pre-computation for encryption...";

      windowsize = 64;
      unsigned long n = 1UL<<(prms.logn); // the dimenssion
      windownum = n/64;

      //      if (windownum > 128){windowsize = 128;windownum=n/64;}      

      uint32_t* pwrFac,*pwrFixFac;
      uint32_t rLength;

      uint32_t* r;

      r=(uint32_t *)malloc(sizeof(uint32_t)*words*windowsize);

      tp = 1;  
      for(int i =0; i < windowsize; i ++)
      {
         BytesFromZZ((unsigned char*)&r[i*words], tp,words*4);
         tp = MulMod(tp, root, det);

      }
      rLength=words*windowsize;

      ZZ tpr64;

      tpr64 = tp;

      pwrFac=(uint32_t *)malloc(sizeof(uint32_t)*words);
      pwrFixFac=(uint32_t *)malloc(sizeof(uint32_t)*words);
      ZZ tpone;
      tpone = 1;
      BytesFromZZ((unsigned char*)pwrFac, tpone,words*4);
      BytesFromZZ((unsigned char*)pwrFixFac, tp,words*4);

      encrypt_FHE_pre(&gpuPwr,&gpuPwr64, samples, gpuU, words, gpuN, dLength+5, q, r, rLength, pwrFac, pwrFixFac,windowsize,windownum);

      free(pwrFac);
      free(pwrFixFac);
      free(r);

      inited1 = true;
      cout<<"done"<<endl;
      getFreeMemory() ;
   }   
   return true;

}

bool FHEkeys::gpuPreComputation2()
{
   if(inited2)return false;
   if(!inited0)return false;

   getFreeMemory() ;
   cout<<"Pre-computation for recrypt...";
   uint32_t *ctxts_bytes;
   uint32_t   *gpuCtxts;

   unsigned long nCtxts = mChoose2(prms.S);

   ctxts_bytes = (uint32_t*)malloc(4*words*nCtxts*prms.s);

   for(int i = 0; i < nCtxts*prms.s; i ++)
   {
      BytesFromZZ((unsigned char*)(ctxts_bytes + words*i), ctxts[i],words*4);

   }
   //gpuCtxts = gpuPush32(ctxts_bytes, words*nCtxts*prms.s);

   gpuCtxtsFFT = (uint2**)malloc(words*nCtxts*prms.s*sizeof(uint2*));
   gpuRecryptPre(gpuCtxtsFFT,ctxts_bytes);

   free(ctxts_bytes);

   inited2 = true;

   cout<<"done"<<endl;

   getFreeMemory() ;

   return true;

}

bool FHEkeys::gpuEncrypt(vec_ZZ& c, unsigned int b[], int num) const
{
   int i;
   ZZ tmp;       // used to hold r^m in the recursive calls
   unsigned long n = 1UL<<(prms.logn); // the dimenssion
   double p = ((double)prms.noise)/n;  // # of expected nonzero coefficients
   if (p>0.5) p = 0.5;
   c.SetLength(num);
   // Evaluate all the polynomials together at root mode det

   // Set c[i] = 2*c[i] + b[i]

   uint32_t* z;

   z=(uint32_t *)malloc(sizeof(uint32_t)*words);

   for (i=0; i<num; i++) {
  
      uint32_t s[n];
      for(int j=0; j<n; j++) s[j]=randomBit(p);

      encrypt_FHE_FFT(z, samples, gpuU, words, gpuN, dLength+5, s, q,gpuPwr,gpuPwr64,windowsize,windownum);

      c[i] = ZZFromBytes((unsigned char*)z, 4*words);

      c[i] <<= 1;
      c[i] += b[i];
      if (c[i]>=det) c[i] -= det;
   }

   free(z);



   return true;
}
bool FHEkeys::ReleasePreEnc()
{
   if(!inited1)return false;
   inited1 = false;
   if(windownum < 64){   $GPU(cudaFree(gpuPwr));}
   else free(gpuPwr);

   if(windownum < 64){  $GPU(cudaFree(gpuPwr64))}
   else free(gpuPwr64);
};



bool FHEkeys::ReleasePreRecrypt()
{
   if(!inited2)return false;
   inited2 = false;
   unsigned long nCtxts = mChoose2(prms.S);
   for(int i = 0; i < nCtxts*prms.s; i ++){
      $GPU(cudaFree(gpuCtxtsFFT[i]));
   }
   free(gpuCtxtsFFT);


};

bool FHEkeys::Release()
{

   $GPU(cudaFree(gpuU));
   $GPU(cudaFree(gpuN));
   inited0 = false;
   ReleasePreEnc();

   ReleasePreRecrypt();
   return true;
}


