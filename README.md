基于 CUDA 实现的 SM4-CTR 模式加解密，并对计算结果与 GMSSL 库进行了验证对比。

该实现参考了[SM4-CTR](https://github.com/BESTICSP/SM4-CTR)和[GmSSL](https://github.com/guanzhi/GmSSL)的实现，将计算结果和GmSSL进行了校验，以保证计算结果的正确性。

Based on the CUDA implementation of the SM4-CTR mode for encryption and decryption, the calculation results were verified and compared with the GmSSL library.

This implementation referenced the approaches from [SM4-CTR](https://github.com/BESTICSP/SM4-CTR) and [GmSSL](https://github.com/guanzhi/GmSSL), and the results were cross-checked with GmSSL to ensure correctness.
