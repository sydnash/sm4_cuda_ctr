#ifndef __sm4_cuda__h__
#define __sm4_cuda__h__

#include <stdint.h>

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG 1

#define error_print()                                            \
    do {                                                         \
        if (DEBUG)                                               \
            fprintf(stderr, "%s:%d:%s():\n", __FILE__, __LINE__, \
                    __FUNCTION__);                               \
    } while (0)

#define checkCudaError(result, message)                                    \
    do {                                                                   \
        cudaError_t __err__ = (result);                                    \
        if (__err__ != cudaSuccess) {                                      \
            error_print();                                                 \
            printf("%s  (Error code: %d - %s\n)", (message), (int)__err__, \
                   cudaGetErrorString(__err__));                           \
            exit(-1);                                                      \
        }                                                                  \
    } while (0)


#define SM4_KEY_SIZE (16)
#define SM4_BLOCK_SIZE (16)
#define SM4_NUM_ROUNDS (32)

void sm4_cpu_ctr(
        uint8_t* text, uint8_t* cipher, size_t size, uint32_t* key,
        uint8_t* ctr);

void sm4_gpu_ctr(
        uint8_t* text, uint8_t* cipher, size_t size, uint8_t* raw_key,
        uint8_t* ctr, int device_id, size_t max_thread);

typedef struct {
    uint32_t rk[SM4_NUM_ROUNDS];
} sm4_key_t;

void sm4_cuda_set_encrypt_key(
        sm4_key_t* key, const uint8_t raw_key[SM4_KEY_SIZE]);
void sm4_cuda_set_decrypt_key(
        sm4_key_t* key, const uint8_t raw_key[SM4_KEY_SIZE]);

typedef struct {
    uint8_t initialized;
    sm4_key_t sm4_key;
    uint8_t left_data[SM4_BLOCK_SIZE];
    size_t left_data_size;

    // cuda param
    int device_id;
    size_t max_thread;
    size_t max_in_output_size;
    size_t max_thread_per_block;
    cudaStream_t stream;
    uint32_t* device_rk;
    uint8_t* device_counter;
    uint8_t* device_input;
    uint8_t* device_output;

} sm4_ctr_ctx_t;

int sm4_ctr32_cuda_encrypt_init(
        sm4_ctr_ctx_t* ctx, const uint8_t key[SM4_KEY_SIZE],
        const uint8_t ctr[SM4_BLOCK_SIZE], int device_id, size_t max_thread);
// pass out as NULL to get the correct out memory size.
int sm4_ctr32_cuda_encrypt_update(
        sm4_ctr_ctx_t* ctx, const uint8_t* in, size_t inlen, uint8_t* out,
        size_t* outlen);
int sm4_ctr32_cuda_encrypt_finish(
        sm4_ctr_ctx_t* ctx, uint8_t* out, size_t* outlen);

#ifdef __cplusplus
}
#endif

#endif