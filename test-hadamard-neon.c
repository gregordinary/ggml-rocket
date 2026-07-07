// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-hadamard-neon.c — test for the NEON Hadamard rotation.
 *
 * A scalar Hadamard rotation dominates the W8A8 cost, so ggml-rocket.cpp
 * NEON-vectorizes it. This compares the NEON fp32 rotation (a MIRROR of
 * ggml-rocket.cpp's rk_hadamard_rotate) against a scalar reference for every
 * Gemma K, and checks the rotation is orthonormal (norm-preserving). Catches NEON
 * algorithm bugs in <1s, before any 22GB model load.
 *
 * On a non-NEON host both paths are scalar (trivial pass = build check only); the
 * real check runs on the device (aarch64). The end-to-end check is still
 * ROCKET_VERIFY=1 in-model (product-preservation per op).
 *
 * Build (device):  gcc -O3 -march=native -o /tmp/test_had test-hadamard-neon.c -lm
 * Run:             /tmp/test_had
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define HAVE_NEON 1
#endif

/* ---- shared H_60 construction (Paley I, q=59) ---- */
static int  rk_modpow(long a, long e, long p){ long r=1; a%=p; while(e){ if(e&1) r=r*a%p; a=a*a%p; e>>=1; } return (int)r; }
static int  rk_legendre(int a, int p){ a%=p; if(a<0)a+=p; if(a==0)return 0; return rk_modpow(a,(p-1)/2,p)==1?1:-1; }
static float g_H60[60][60]; static bool g_H60_ready=false;
static void rk_build_H60(void){ if(g_H60_ready)return; const int q=59;
    for(int i=0;i<60;i++) for(int j=0;j<60;j++){
        if(i==0)      g_H60[i][j]= 1;
        else if(j==0) g_H60[i][j]=-1;
        else if(i==j) g_H60[i][j]= 1;
        else          g_H60[i][j]=(float)rk_legendre((i-1)-(j-1),q); }
    g_H60_ready=true; }
static bool rk_ispow2(int x){ return x>0 && (x&(x-1))==0; }

/* ---- scalar reference (golden) ---- */
static void fwht_ref(float *v,int n){
    for(int len=1;len<n;len<<=1) for(int i=0;i<n;i+=len<<1)
        for(int j=i;j<i+len;j++){ float a=v[j],b=v[j+len]; v[j]=a+b; v[j+len]=a-b; } }
static void rotate_ref(float *row,int K){ const float inv=1.0f/sqrtf((float)K);
    if(rk_ispow2(K)) fwht_ref(row,K);
    else { const int B2=K/60; float col[4096];
        for(int b=0;b<B2;b++){ float *blk=row+b*60,out[60];
            for(int o=0;o<60;o++){ float s=0; for(int k=0;k<60;k++) s+=g_H60[o][k]*blk[k]; out[o]=s; }
            for(int o=0;o<60;o++) blk[o]=out[o]; }
        for(int lane=0;lane<60;lane++){ for(int b=0;b<B2;b++) col[b]=row[b*60+lane];
            fwht_ref(col,B2); for(int b=0;b<B2;b++) row[b*60+lane]=col[b]; } }
    for(int k=0;k<K;k++) row[k]*=inv; }

/* ---- version under test: NEON where available (mirror of ggml-rocket.cpp) ---- */
static void fwht_simd(float *v,int n){
    for(int len=1;len<n;len<<=1) for(int i=0;i<n;i+=len<<1){ int j=i;
#ifdef HAVE_NEON
        if(len>=4) for(;j+4<=i+len;j+=4){ float32x4_t a=vld1q_f32(v+j),b=vld1q_f32(v+j+len);
            vst1q_f32(v+j,vaddq_f32(a,b)); vst1q_f32(v+j+len,vsubq_f32(a,b)); }
#endif
        for(;j<i+len;j++){ float a=v[j],b=v[j+len]; v[j]=a+b; v[j+len]=a-b; } } }
static void rotate_simd(float *row,int K){ const float inv=1.0f/sqrtf((float)K);
    if(rk_ispow2(K)) fwht_simd(row,K);
    else { const int B2=K/60; float col[4096];
        for(int b=0;b<B2;b++){ float *blk=row+b*60,out[60];
            for(int o=0;o<60;o++){ const float *h=g_H60[o];
#ifdef HAVE_NEON
                float32x4_t acc=vdupq_n_f32(0.0f);
                for(int k=0;k<60;k+=4) acc=vfmaq_f32(acc,vld1q_f32(h+k),vld1q_f32(blk+k));
                out[o]=vaddvq_f32(acc);
#else
                float s=0; for(int k=0;k<60;k++) s+=h[k]*blk[k]; out[o]=s;
#endif
            }
            for(int o=0;o<60;o++) blk[o]=out[o]; }
        for(int lane=0;lane<60;lane++){ for(int b=0;b<B2;b++) col[b]=row[b*60+lane];
            fwht_simd(col,B2); for(int b=0;b<B2;b++) row[b*60+lane]=col[b]; } }
    int k=0;
#ifdef HAVE_NEON
    const float32x4_t vinv=vdupq_n_f32(inv);
    for(;k+4<=K;k+=4) vst1q_f32(row+k,vmulq_f32(vld1q_f32(row+k),vinv));
#endif
    for(;k<K;k++) row[k]*=inv; }

int main(void){
    rk_build_H60();
#ifdef HAVE_NEON
    printf("NEON: ENABLED -> real scalar-vs-NEON gate\n");
#else
    printf("NEON: unavailable on this host -> scalar-vs-scalar (build check only)\n");
#endif
    const int Ks[]={512,2048,3840,4096,8192,15360};
    int fails=0;
    for(int t=0;t<6;t++){ int K=Ks[t];
        float *r=malloc(K*sizeof(float)), *s=malloc(K*sizeof(float));
        srand(1234+K);
        double n0=0;
        for(int i=0;i<K;i++){ float v=((float)rand()/(float)RAND_MAX-0.5f)*20.0f; r[i]=v; s[i]=v; n0+=(double)v*v; }
        rotate_ref(r,K); rotate_simd(s,K);
        double md=0,nr=0;
        for(int i=0;i<K;i++){ double d=fabs((double)r[i]-(double)s[i]); if(d>md)md=d; nr+=(double)s[i]*s[i]; }
        double normerr=fabs(sqrt(nr)-sqrt(n0))/sqrt(n0);   /* orthonormal: ||Hx||==||x|| */
        bool ok = (md < 1e-2) && (normerr < 1e-3);
        printf("K=%-6d  max|neon-ref|=%.3e  norm-preserve-err=%.3e  %s\n",
               K, md, normerr, ok?"PASS":"FAIL");
        if(!ok) fails++;
        free(r); free(s);
    }
    printf("%s\n", fails?"FAILED":"ALL PASS");
#ifdef HAVE_NEON
    return fails?1:0;
#else
    // The code under test (the NEON kernel) was NOT exercised here: both paths are the
    // scalar reference, so a "pass" only confirms the build + the rotation's
    // orthonormality. Report a real failure if THAT self-check broke; otherwise SKIP
    // (77 = autotools/CTest convention) so a non-aarch64 run can't masquerade as a
    // NEON validation. The real gate runs on the device (aarch64).
    if (fails) return 1;
    printf("SKIP: NEON path not exercised on this host (scalar build/self-check only; run on aarch64)\n");
    return 77;
#endif
}
