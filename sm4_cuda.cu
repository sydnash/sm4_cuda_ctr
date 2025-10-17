#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm4_cuda.h"


// sbox for host and device.
#ifdef __CUDA_ARCH__
__constant__
#endif
        uint8_t sbox[256] = {
                0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6,
                0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05, 0x2b, 0x67, 0x9a, 0x76,
                0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86,
                0x06, 0x99, 0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a,
                0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62, 0xe4, 0xb3,
                0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa,
                0x75, 0x8f, 0x3f, 0xa6, 0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73,
                0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
                0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb,
                0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35, 0x1e, 0x24, 0x0e, 0x5e,
                0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21,
                0x78, 0x87, 0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52,
                0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e, 0xea, 0xbf,
                0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce,
                0xf9, 0x61, 0x15, 0xa1, 0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34,
                0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
                0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29,
                0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f, 0xd5, 0xdb, 0x37, 0x45,
                0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c,
                0x5b, 0x51, 0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f,
                0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8, 0x0a, 0xc1,
                0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12,
                0xb8, 0xe5, 0xb4, 0xb0, 0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96,
                0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
                0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee,
                0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48};

/*********************ck**********************/
const uint32_t ck[32] = {
        0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269, 0x70777e85, 0x8c939aa1,
        0xa8afb6bd, 0xc4cbd2d9, 0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
        0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9, 0xc0c7ced5, 0xdce3eaf1,
        0xf8ff060d, 0x141b2229, 0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
        0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209, 0x10171e25, 0x2c333a41,
        0x484f565d, 0x646b7279};

#define ROTL(_x, _y) (((_x) << (_y)) | (((_x) & 0xffffffff) >> (32 - (_y))))
#define L1(_B) ((_B) ^ ROTL(_B, 2) ^ ROTL(_B, 10) ^ ROTL(_B, 18) ^ ROTL(_B, 24))
#define L2(_B) ((_B) ^ ROTL(_B, 13) ^ ROTL(_B, 23))
#define S_BOX(_A)                                                        \
    ((sbox[(_A) >> 24 & 0xFF] << 24) | (sbox[(_A) >> 16 & 0xFF] << 16) | \
     (sbox[(_A) >> 8 & 0xFF] << 8) | (sbox[(_A) & 0xFF]))
//#define T(_A) ( \
//    L1(Sbox[(_A) >> 24 & 0xFF] << 24) ^ \
//    L1(Sbox[(_A) >> 16 & 0xFF] << 16) ^ \
//    L1(Sbox[(_A) >>  8 & 0xFF] <<  8) ^ \
//    L1(Sbox[(_A) & 0xFF]))
#define T(_A) L1(S_BOX(_A))
#define T2(_A) L2(S_BOX(_A))


// big endine
#define PUTU32(p, V)                                               \
    ((p)[0] = (uint8_t)((V) >> 24), (p)[1] = (uint8_t)((V) >> 16), \
     (p)[2] = (uint8_t)((V) >> 8), (p)[3] = (uint8_t)(V))

#define GETU32(p)                                              \
    (((uint32_t)((p)[0])) << 24 | ((uint32_t)((p)[1])) << 16 | \
     ((uint32_t)((p)[2])) << 8 | ((uint32_t)((p)[3])))

void
sm4_key_extend(const uint8_t* key, uint32_t* rk, uint32_t decrypt)
{
    uint32_t r, mid, x0, x1, x2, x3;
    x0 = GETU32(key);
    x1 = GETU32(key + 4);
    x2 = GETU32(key + 8);
    x3 = GETU32(key + 12);
    x0 ^= 0xa3b1bac6;
    x1 ^= 0x56aa3350;
    x2 ^= 0x677d9197;
    x3 ^= 0xb27022dc;
    for (r = 0; r < 32; r += 4) {
        mid = x1 ^ x2 ^ x3 ^ ck[r];
        rk[r] = x0 ^= T2(mid);
        mid = x2 ^ x3 ^ x0 ^ ck[r + 1];
        rk[r + 1] = x1 ^= T2(mid);
        mid = x3 ^ x0 ^ x1 ^ ck[r + 2];
        rk[r + 2] = x2 ^= T2(mid);
        mid = x0 ^ x1 ^ x2 ^ ck[r + 3];
        rk[r + 3] = x3 ^= T2(mid);
    }
    // decrypt, reverse the rk order.
    if (decrypt == 1) {
        for (r = 0; r < 16; r++) {
            mid = rk[r], rk[r] = rk[31 - r], rk[31 - r] = mid;
        }
    }
}

__host__ __device__ void
sm4_crypt(
        const uint8_t* input, uint8_t* output, uint32_t* rk, int flag,
        const uint8_t* plaintext)
{
    uint32_t r, mid, x0, x1, x2, x3;

    // get input as for big endine 32 bit number
    x0 = GETU32(input);
    x1 = GETU32(input + 4);
    x2 = GETU32(input + 8);
    x3 = GETU32(input + 12);

    //
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (r = 0; r < 32; r += 4) {
        // 利用轮密钥，加解密
        mid = x1 ^ x2 ^ x3 ^ rk[r + 0];
        x0 ^= T(mid);
        mid = x2 ^ x3 ^ x0 ^ rk[r + 1];
        x1 ^= T(mid);
        mid = x3 ^ x0 ^ x1 ^ rk[r + 2];
        x2 ^= T(mid);
        mid = x0 ^ x1 ^ x2 ^ rk[r + 3];
        x3 ^= T(mid);
    }

    // if mode="CTR"
    if (flag == 1) {
        x3 ^= GETU32(plaintext);
        x2 ^= GETU32(plaintext + 4);
        x1 ^= GETU32(plaintext + 8);
        x0 ^= GETU32(plaintext + 12);
    }

    // put output
    PUTU32(output, x3);
    PUTU32(output + 4, x2);
    PUTU32(output + 8, x1);
    PUTU32(output + 12, x0);
}

__global__ void
sm4_encrypt_cuda_ctr(
        uint8_t* input, uint8_t* output, uint32_t* rk,
        int64_t num_cipher_blocks, uint8_t* counter)
{
    int64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < num_cipher_blocks) {
        uint8_t ca[16];
        for (int w = 0; w < 16; w++) {
            ca[w] = counter[w];
        }
        uint32_t cc = GETU32(&(ca[12]));
        cc += i;
        PUTU32(&(ca[12]), cc);
        sm4_crypt(&(ca[0]), output + i * 16, rk, 1, input + i * 16);
        // last block, increase counter by one and write counter back to input
        // for next update.
        if (i == num_cipher_blocks - 1) {
            cc += 1;
            PUTU32(&(ca[12]), cc);
            for (int w = 0; w < 16; w++) {
                counter[w] = ca[w];
            }
        }
    }
}

static void
ctr32_incr(uint8_t a[SM4_BLOCK_SIZE])
{
    int i;
    for (i = 15; i >= 12; i--) {
        a[i]++;
        if (a[i])
            break;
    }
}

void
sm4_cpu_ctr(
        uint8_t* text, uint8_t* cipher, size_t size, uint32_t* key,
        uint8_t* ctr)
{
    uint8_t* p = (uint8_t*)malloc(size * sizeof(uint8_t));
    memcpy(p, text, size * sizeof(uint8_t));
    uint8_t* c = (uint8_t*)malloc(size * sizeof(uint8_t));
    uint8_t* counter = (uint8_t*)malloc(16 * sizeof(uint8_t));
    memcpy(counter, ctr, 16 * sizeof(uint8_t));

    uint32_t* rk = (uint32_t*)malloc(SM4_NUM_ROUNDS * sizeof(uint32_t));
    memcpy(rk, key, SM4_NUM_ROUNDS * sizeof(uint32_t));

    size_t n = size >> 4;

    uint8_t ca[SM4_BLOCK_SIZE];
    for (int w = 0; w < SM4_BLOCK_SIZE; w++) {
        ca[w] = counter[w];
    }
    for (size_t i = 0; i < n; i++) {
        sm4_crypt(ca, c + i * 16, rk, 1, p + i * 16);
        ctr32_incr(ca);
    }

    memcpy(cipher, c, size * sizeof(uint8_t));

    free(c);
    free(p);
    free(counter);
    free(rk);
}

static void
freeDeviceMemory(void** p, const char* message)
{
    checkCudaError(cudaFree(*p), message);
    *p = NULL;
}

#ifdef __cplusplus
extern "C" {
#endif

void
sm4_cuda_set_encrypt_key(sm4_key_t* key, const uint8_t raw_key[SM4_KEY_SIZE])
{
    sm4_key_extend(&raw_key[0], &key->rk[0], 0);
}

void
sm4_cuda_set_decrypt_key(sm4_key_t* key, const uint8_t raw_key[SM4_KEY_SIZE])
{
    sm4_key_extend(&raw_key[0], &key->rk[0], 1);
}

int
sm4_ctr32_cuda_encrypt_init(
        sm4_ctr_ctx_t* ctx, const uint8_t key[SM4_KEY_SIZE],
        const uint8_t ctr[SM4_BLOCK_SIZE], int device_id, size_t max_thread)
{
    struct cudaDeviceProp prop;
    memset(ctx, 0, sizeof(*ctx));

    sm4_cuda_set_encrypt_key(&ctx->sm4_key, key);
    ctx->left_data_size = 0;

    ctx->max_thread = max_thread;
    ctx->device_id = device_id;
    checkCudaError(cudaSetDevice(device_id), "set device id");

    size_t max_in_output_size = SM4_BLOCK_SIZE * max_thread;
    ctx->max_in_output_size = max_in_output_size;
    checkCudaError(cudaGetDeviceProperties(&prop, 0), "get device properties");
    ctx->max_thread_per_block = prop.maxThreadsPerBlock;
    checkCudaError(cudaStreamCreate(&ctx->stream), "create stream");
    checkCudaError(
            cudaMalloc(&ctx->device_rk, sizeof(ctx->sm4_key)),
            "alloc device rk");
    checkCudaError(
            cudaMemcpy(
                    ctx->device_rk, &ctx->sm4_key, sizeof(ctx->sm4_key),
                    cudaMemcpyHostToDevice),
            "copy rk");
    checkCudaError(
            cudaMalloc(&ctx->device_counter, SM4_BLOCK_SIZE),
            "alloc device counter");
    checkCudaError(
            cudaMemcpy(
                    ctx->device_counter, ctr, SM4_BLOCK_SIZE,
                    cudaMemcpyHostToDevice),
            "copy counter to device");
    checkCudaError(
            cudaMalloc(&ctx->device_input, max_in_output_size),
            "alloc input memory");
    checkCudaError(
            cudaMalloc(&ctx->device_output, max_in_output_size),
            "alloc output memory");

    ctx->initialized = 1;
    return 0;
}

int
sm4_ctr32_cuda_encrypt_update(
        sm4_ctr_ctx_t* ctx, const uint8_t* in, size_t inlen, uint8_t* out,
        size_t* outlen)
{
    size_t total_size, cur_size, left, write_pos = 0, read_pos = 0;
    if (!ctx || !ctx->initialized || !in || !outlen) {
        error_print();
        return -1;
    }
    if (!out) {  // out is NULL, return the expect out memory size.
        *outlen = 16 * ((inlen + ctx->left_data_size + 15) / 16);
        return 0;
    }
    if (ctx->left_data_size >= SM4_BLOCK_SIZE) {
        error_print();
        return -1;
    }

    total_size = inlen + ctx->left_data_size;
    if (total_size < SM4_BLOCK_SIZE) {
        memcpy(ctx->left_data + ctx->left_data_size, in, inlen);
        ctx->left_data_size += inlen;
        *outlen = 0;
        return 0;
    }

    *outlen = 0;

    checkCudaError(cudaSetDevice(ctx->device_id), "set device id");

    while (total_size >= SM4_BLOCK_SIZE) {
        write_pos = 0;
        cur_size =
                min(ctx->max_in_output_size,
                    total_size / SM4_BLOCK_SIZE * SM4_BLOCK_SIZE);
        total_size -= cur_size;
        left = cur_size;
        if (ctx->left_data_size > 0) {
            cudaMemcpyAsync(
                    ctx->device_input + write_pos, ctx->left_data,
                    ctx->left_data_size, cudaMemcpyHostToDevice, ctx->stream);
            write_pos += ctx->left_data_size;
            left -= ctx->left_data_size;
            ctx->left_data_size = 0;
        }
        cudaMemcpyAsync(
                ctx->device_input + write_pos, in + read_pos, left,
                cudaMemcpyHostToDevice, ctx->stream);
        read_pos += left;

        size_t threads, blocks, n;
        n = cur_size / SM4_BLOCK_SIZE;
        assert(n <= ctx->max_thread);
        threads =
                (n < ctx->max_thread_per_block) ? n : ctx->max_thread_per_block;
        blocks = (n + threads - 1) / threads;
        sm4_encrypt_cuda_ctr<<<blocks, threads, 0, ctx->stream>>>(
                ctx->device_input, ctx->device_output, ctx->device_rk, n,
                ctx->device_counter);

        cudaMemcpyAsync(
                out + *outlen, ctx->device_output, cur_size,
                cudaMemcpyDeviceToHost, ctx->stream);
        cudaStreamSynchronize(ctx->stream);

        *outlen += cur_size;
    }
    if (total_size > 0) {
        ctx->left_data_size = total_size;
        memcpy(ctx->left_data, in + read_pos, ctx->left_data_size);
    }

    return 0;
}

int
sm4_ctr32_cuda_encrypt_finish(sm4_ctr_ctx_t* ctx, uint8_t* out, size_t* outlen)
{
    if (!ctx || !ctx->initialized || !outlen) {
        error_print();
        return -1;
    }
    if (!out) {
        *outlen = ctx->left_data_size;
        return 0;
    }
    if (ctx->left_data_size >= SM4_BLOCK_SIZE) {
        error_print();
        return -1;
    }
    if (ctx->left_data_size > 0) {
        cudaMemcpyAsync(
                ctx->device_input, ctx->left_data, ctx->left_data_size,
                cudaMemcpyHostToDevice, ctx->stream);
        sm4_encrypt_cuda_ctr<<<1, 1, 0, ctx->stream>>>(
                ctx->device_input, ctx->device_output, ctx->device_rk, 1,
                ctx->device_counter);
        cudaMemcpyAsync(
                out, ctx->device_output, ctx->left_data_size,
                cudaMemcpyDeviceToHost, ctx->stream);
        cudaStreamSynchronize(ctx->stream);
    }
    *outlen = ctx->left_data_size;
    ctx->left_data_size = 0;

    freeDeviceMemory((void**)&ctx->device_rk, "free device rk");
    freeDeviceMemory((void**)&ctx->device_counter, "free device counter");
    freeDeviceMemory((void**)&ctx->device_input, "free device input");
    freeDeviceMemory((void**)&ctx->device_output, "free device output");

    ctx->initialized = 0;
    return 0;
}

void
sm4_gpu_ctr(
        uint8_t* text, uint8_t* cipher, size_t size, uint8_t* raw_key,
        uint8_t* ctr, int device_id, size_t max_thread)
{
    checkCudaError(cudaSetDevice(device_id), "set device id");
    sm4_ctr_ctx_t ctx;
    if (sm4_ctr32_cuda_encrypt_init(&ctx, raw_key, ctr, 0, max_thread) != 0) {
        error_print();
        return;
    }

    size_t input_pos = 0, out_pos = 0;
    size_t output_len = 0;
    while (size > 0) {
        size_t input_len =
                size < ctx.max_in_output_size ? size : ctx.max_in_output_size;
        size -= input_len;

        uint8_t* in = text + input_pos;
        uint8_t* out = cipher + out_pos;
        if (sm4_ctr32_cuda_encrypt_update(
                    &ctx, in, input_len, out, &output_len) != 0) {
            error_print();
            return;
        }

        input_pos += input_len;
        out_pos += output_len;
    }
    if (sm4_ctr32_cuda_encrypt_finish(&ctx, cipher + out_pos, &output_len) !=
        0) {
        error_print();
    }
}

#ifdef __cplusplus
}
#endif
