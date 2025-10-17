#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "sm4_cuda.h"

#ifdef GMSSL_TEST
#include "gmssl/sm4.h"
#endif

static uint8_t*
alloc_memory(size_t size)
{
    uint8_t* p = (uint8_t*)malloc(size);
    memset(p, 0, size);
    return p;
}

#ifdef GMSSL_TEST
typedef struct {
    SM4_KEY gm_rk;
    SM4_KEY gm_drk;
} gm_ssl_t;

void
init_sm4_key(gm_ssl_t* gm_ssl, uint8_t key[SM4_KEY_SIZE])
{
    sm4_set_encrypt_key(&gm_ssl->gm_rk, key);
    sm4_set_decrypt_key(&gm_ssl->gm_drk, key);
}

int
check_extend_key(gm_ssl_t* gm_ssl, sm4_key_t encrypt_key, sm4_key_t decrypt_key)
{
    if (memcmp(encrypt_key.rk, gm_ssl->gm_rk.rk, SM4_NUM_ROUNDS) != 0) {
        printf("encrypt key is not same as gm_ssl\n");
        return -1;
    }
    if (memcmp(decrypt_key.rk, gm_ssl->gm_drk.rk, SM4_NUM_ROUNDS) != 0) {
        printf("decrypt key is not same as gm_ssl\n");
        return -1;
    }
    return 0;
}

void
gm_sm4_encrypt(
        gm_ssl_t* gm_ssl, uint8_t* ctr, uint8_t* input, size_t size,
        uint8_t* output)
{
    sm4_ctr32_encrypt(&gm_ssl->gm_rk, ctr, input, size, output);
}

static int
check_correct()
{
    unsigned char key[SM4_KEY_SIZE] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                                       0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98,
                                       0x76, 0x54, 0x32, 0x10};

    sm4_key_t encrypt_key, decrypt_key;
    sm4_cuda_set_encrypt_key(&encrypt_key, &key[0]);  // encrypt
    sm4_cuda_set_decrypt_key(&decrypt_key, &key[0]);  // decrypt

    gm_ssl_t gm_ssl;
    init_sm4_key(&gm_ssl, key);
    if (check_extend_key(&gm_ssl, encrypt_key, decrypt_key)) {
        return -1;
    }

    unsigned char counter[SM4_BLOCK_SIZE] = "abftnbtfreskjuy";
#define TEST_SIZE SM4_BLOCK_SIZE * 1024 * 1024UL
    srand(time(NULL));
    uint8_t* data = alloc_memory(TEST_SIZE);
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        data[i] = rand() % 256;
    }
    uint8_t* cpu_encrypt_data = alloc_memory(TEST_SIZE);
    uint8_t* cpu_decrypt_data = alloc_memory(TEST_SIZE);
    uint8_t* gm_encrypt_data = alloc_memory(TEST_SIZE);
    uint8_t* gm_decrypt_data = alloc_memory(TEST_SIZE);

    sm4_cpu_ctr(data, cpu_encrypt_data, TEST_SIZE, encrypt_key.rk, counter);
    sm4_cpu_ctr(
            cpu_encrypt_data, cpu_decrypt_data, TEST_SIZE, encrypt_key.rk,
            counter);
    if (memcmp(cpu_decrypt_data, data, TEST_SIZE) != 0) {
        printf("cpu decrypt failed\n");
        return -1;
    } else {
        printf("cpu decrypt ok\n");
    }

    // sm4_ctr32_encrpt will change the Counter
    uint8_t* ctr = alloc_memory(sizeof(counter));
    memcpy(ctr, counter, sizeof(counter));
    gm_sm4_encrypt(&gm_ssl, ctr, data, TEST_SIZE, gm_encrypt_data);
    if (memcmp(cpu_encrypt_data, gm_encrypt_data, TEST_SIZE) != 0) {
        printf("cpu encrypt data is not same as gm_ssl ctr32_encrypt\n");
        return -1;
    } else {
        printf("cpu encrypt ok\n");
    }

    sm4_ctr_ctx_t ctx;
    sm4_ctr32_cuda_encrypt_init(&ctx, key, counter, 0, 1024);

    size_t read_pos = 0;
    size_t out_pos = 0;
    size_t outlen = 0;
    printf("input range: %p %p size: %ld\n", data, data + TEST_SIZE, TEST_SIZE);
    printf("output range: %p %p\n", cpu_encrypt_data,
           cpu_encrypt_data + TEST_SIZE);
    while (read_pos < TEST_SIZE) {
        size_t cur_len = rand() % 1024 + 1024;
        cur_len =
                cur_len < TEST_SIZE - read_pos ? cur_len : TEST_SIZE - read_pos;
        if (sm4_ctr32_cuda_encrypt_update(
                    &ctx, data + read_pos, cur_len, cpu_encrypt_data + out_pos,
                    &outlen)) {
            error_print();
            return -1;
        }
        read_pos += cur_len;
        out_pos += outlen;
    }

    if (sm4_ctr32_cuda_encrypt_finish(
                &ctx, cpu_encrypt_data + out_pos, &outlen)) {
        error_print();
        return -1;
    }

    if (memcmp(cpu_encrypt_data, gm_encrypt_data, TEST_SIZE) != 0) {
        error_print();
        printf("gpu encrypt data is not same as gm_ssl ctr32_encrypt\n");
        return -1;
    } else {
        printf("gpu encrypt ok\n");
    }

    sm4_gpu_ctr(data, cpu_encrypt_data, TEST_SIZE, key, counter, 0, 1024 * 16);
    if (memcmp(cpu_encrypt_data, gm_encrypt_data, TEST_SIZE) != 0) {
        error_print();
        printf("gpu encrypt data is not same as gm_ssl ctr32_encrypt\n");
        return -1;
    } else {
        printf("sm4_gpu_ctr encrypt ok\n");
    }
    sm4_gpu_ctr(
            cpu_encrypt_data, cpu_decrypt_data, TEST_SIZE, key, counter, 0,
            1024 * 16);
    if (memcmp(cpu_decrypt_data, data, TEST_SIZE) != 0) {
        error_print();
        printf("gpu decrypt data is not same as origin data\n");
        return -1;
    } else {
        printf("sm4_gpu_ctr decrypt ok\n");
    }

    free(data);
    free(cpu_encrypt_data);
    free(gm_encrypt_data);
    free(cpu_decrypt_data);
    free(gm_decrypt_data);
    return 0;
}
#endif

double
time_elapse_microseconds(struct timeval start, struct timeval end)
{
    int64_t sec = end.tv_sec - start.tv_sec;
    int64_t mic = end.tv_usec - start.tv_usec;
    return sec * 1000.0 * 1000.0 + mic;
}


int
main(int argc, char** argv)
{
#ifdef GMSSL_TEST
    if (check_correct()) {
        return -1;
    }
#endif

    char* filename;
    if (argc > 1) {
        filename = argv[1];
    } else {
        printf("no filename specified, if you want to test speed, please "
               "use\n");
        printf("\t%s filename to test speed. or\n", argv[0]);
        printf("\t%s filename threadcount to test speed. or\n", argv[0]);
        printf("\tfilename is the file name that contain test data\n");
        printf("\tthreadcount is the max cuda thread on device used to encrypt "
               "data, such as 100 100M 1G\n");
        return 0;
    }

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("open file %s failed!\n", filename);
        return -1;
    }
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);
    printf("test speed size: %ld\n", size);
    uint8_t* p = (uint8_t*)calloc(size, sizeof(uint8_t));
    fread(p, size, 1, file);
    fclose(file);

    printf("明文大小:%.4fMB\n", size / (1024.f * 1024.f));

    uint8_t* encrypt_data = (uint8_t*)calloc(size, sizeof(char));
    uint8_t* decrypt_data = (uint8_t*)calloc(size, sizeof(char));

    unsigned char key[SM4_KEY_SIZE] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                                       0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98,
                                       0x76, 0x54, 0x32, 0x10};
    unsigned char counter[SM4_BLOCK_SIZE] = "abftnbtfreskjuy";

    struct timeval time_start, time_end;
    double time_elapsed;
    size_t thread_count = 1024 * 1024;
    if (argc > 2) {
        char* left_pos = NULL;
        size_t input_value = strtoll(argv[2], &left_pos, 10);
        if (left_pos == argv[2]) {
            printf("invalid input value, use default thread count 1M\n");
            return -1;
        }
        if (*left_pos == 'M') {
            input_value = input_value * 1024 * 1024;
        } else if (*left_pos == 'G') {
            input_value = input_value * 1024 * 1024 * 1024;
        }
        thread_count = input_value;
    }
    printf("device max thread count is %.2fM\n",
           (double)thread_count / 1024 / 1024);
    sm4_gpu_ctr(p, encrypt_data, size, key, counter, 0, thread_count);
    gettimeofday(&time_start, NULL);
    sm4_gpu_ctr(p, encrypt_data, size, key, counter, 0, thread_count);
    gettimeofday(&time_end, NULL);
    time_elapsed = time_elapse_microseconds(time_start, time_end);
    printf("ctr encryption takes %1f ms.\n", time_elapsed / 1000);
    printf("speed = %.5fGB/s\n",
           size / 1024.0 / 1024.0 / 1024.0 / (time_elapsed / 1000000.0));

    gettimeofday(&time_start, NULL);
    sm4_gpu_ctr(
            encrypt_data, decrypt_data, size, key, counter, 0, thread_count);
    gettimeofday(&time_end, NULL);
    time_elapsed = time_elapse_microseconds(time_start, time_end);
    printf("ctr decryption takes %1f ms.\n", time_elapsed / 1000);
    printf("speed = %.5fGB/s\n",
           size / 1024.0 / 1024.0 / 1024.0 / (time_elapsed / 1000000.0));

    for (size_t i = 0; i < size; ++i) {
        if (p[i] != decrypt_data[i]) {
            printf("i=%ld, d=%d %d\n", i, p[i], decrypt_data[i]);
        }
    }

    if (memcmp(p, decrypt_data, size) != 0) {
        printf("decrypt data is not same as origin data.\n");
        return -1;
    } else {
        printf("decrypt data is same as origin data, it's ok\n");
    }

    return 0;
}