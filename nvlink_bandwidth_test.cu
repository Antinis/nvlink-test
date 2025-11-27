#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\\n", cudaGetErrorString(err__),    \
                         __FILE__, __LINE__);                                               \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (0)

constexpr int kBlockSize = 256;

__global__ void fill_matrix(float *data, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        data[i] = value;
    }
}

__global__ void peer_sum_kernel(const float *__restrict__ peer_data,
                                size_t n,
                                float *__restrict__ thread_sums) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float local = 0.0f;

    for (size_t i = idx; i < n; i += stride) {
#if __CUDA_ARCH__ >= 350
        local += __ldg(peer_data + i);
#else
        local += peer_data[i];
#endif
    }

    thread_sums[idx] = local;
}

double reduce_on_host(const float *data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(data[i]);
    }
    return sum;
}

int main(int argc, char **argv) {
    size_t rows = 16384;
    size_t cols = 16384;
    if (argc >= 2) {
        rows = std::stoul(argv[1]);
    }
    if (argc >= 3) {
        cols = std::stoul(argv[2]);
    }
    const size_t num_elements = rows * cols;
    const size_t bytes = num_elements * sizeof(float);
    if (num_elements == 0) {
        std::fprintf(stderr, "Matrix size must be > 0\\n");
        return EXIT_FAILURE;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::fprintf(stderr, "Need at least 2 GPUs, only %d found\\n", device_count);
        return EXIT_FAILURE;
    }

    const int compute_device = 0;
    const int data_device = 1;

    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, compute_device, data_device));
    if (!can_access) {
        std::fprintf(stderr, "Device %d cannot access peer %d\\n", compute_device, data_device);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, data_device, compute_device));
    if (!can_access) {
        std::fprintf(stderr, "Device %d cannot be accessed by peer %d\\n", data_device, compute_device);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(compute_device));
    cudaError_t peer_status = cudaDeviceEnablePeerAccess(data_device, 0);
    if (peer_status != cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_CHECK(peer_status);
    } else {
        cudaGetLastError();
    }

    CUDA_CHECK(cudaSetDevice(data_device));
    peer_status = cudaDeviceEnablePeerAccess(compute_device, 0);
    if (peer_status != cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_CHECK(peer_status);
    } else {
        cudaGetLastError();
    }

    CUDA_CHECK(cudaSetDevice(data_device));
    float *remote_matrix = nullptr;
    CUDA_CHECK(cudaMalloc(&remote_matrix, bytes));

    const float init_value = 1.0f;
    int fill_grid = static_cast<int>(std::min((num_elements + kBlockSize - 1) / kBlockSize, static_cast<size_t>(65535)));
    fill_matrix<<<fill_grid, kBlockSize>>>(remote_matrix, num_elements, init_value);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaSetDevice(compute_device));
    int grid = static_cast<int>(std::min((num_elements + kBlockSize - 1) / kBlockSize, static_cast<size_t>(65535)));
    size_t partial_elems = static_cast<size_t>(grid) * kBlockSize;
    float *partial_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&partial_sums, partial_elems * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热一次 kernel，避免冷启动影响计时
    peer_sum_kernel<<<grid, kBlockSize>>>(remote_matrix, num_elements, partial_sums);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    peer_sum_kernel<<<grid, kBlockSize>>>(remote_matrix, num_elements, partial_sums);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msec = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msec, start, stop));

    std::vector<float> host_partial(partial_elems);
    CUDA_CHECK(cudaMemcpy(host_partial.data(), partial_sums,
                          partial_elems * sizeof(float), cudaMemcpyDeviceToHost));

    double total_sum = reduce_on_host(host_partial.data(), partial_elems);
    double expected = static_cast<double>(num_elements) * init_value;
    double bw_gb_per_s = (static_cast<double>(bytes) * 1e-6) / msec;

    std::printf("Matrix: %zu x %zu (%.2f GB)\n", rows, cols, bytes / 1e9);
    std::printf("Kernel time: %.3f ms -> %.2f GB/s effective read\n", msec, bw_gb_per_s);
    std::printf("Computed sum: %.4f (expected %.4f)\n", total_sum, expected);

    CUDA_CHECK(cudaFree(partial_sums));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaSetDevice(data_device));
    CUDA_CHECK(cudaFree(remote_matrix));

    CUDA_CHECK(cudaDeviceReset());
    CUDA_CHECK(cudaSetDevice(compute_device));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

