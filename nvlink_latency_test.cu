#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),     \
                         __FILE__, __LINE__);                                               \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (0)

__global__ void pointer_chase_kernel(const uint32_t *__restrict__ next_indices,
                                     size_t iterations,
                                     uint32_t start,
                                     uint32_t *sink,
                                     unsigned long long *cycles) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t idx = start;
        unsigned long long begin = clock64();
        for (size_t i = 0; i < iterations; ++i) {
#if __CUDA_ARCH__ >= 350
            idx = __ldg(next_indices + idx);
#else
            idx = next_indices[idx];
#endif
        }
        unsigned long long end = clock64();
        sink[0] = idx;            // prevent compiler from optimizing away
        cycles[0] = end - begin;  // cycle count for entire chase
    }
}

struct ChainData {
    std::vector<uint32_t> chain;
    uint32_t start_index;
};

ChainData build_pointer_chain(size_t num_nodes) {
    if (num_nodes < 2) {
        throw std::runtime_error("num_nodes must be >= 2");
    }

    std::vector<uint32_t> perm(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        perm[i] = static_cast<uint32_t>(i);
    }
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<uint32_t> chain(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        uint32_t from = perm[i];
        uint32_t to = perm[(i + 1) % num_nodes];
        chain[from] = to;
    }
    return {std::move(chain), perm[0]};
}

uint32_t *upload_chain_to_device(int device, const std::vector<uint32_t> &chain) {
    CUDA_CHECK(cudaSetDevice(device));
    uint32_t *d_chain = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chain, chain.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_chain, chain.data(), chain.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    return d_chain;
}

struct LatencyResult {
    double avg_cycles_per_iter;
    double avg_ns;
};

LatencyResult measure_latency(int compute_device,
                              const uint32_t *device_chain,
                              size_t num_nodes,
                              size_t iterations,
                              int trials,
                              uint32_t start_index,
                              uint32_t *d_sink,
                              unsigned long long *d_cycles,
                              double clock_rate_khz) {
    CUDA_CHECK(cudaSetDevice(compute_device));
    double total_cycles = 0.0;

    for (int t = 0; t < trials; ++t) {
        pointer_chase_kernel<<<1, 1>>>(device_chain, iterations, start_index,
                                       d_sink, d_cycles);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long host_cycles = 0;
        CUDA_CHECK(cudaMemcpy(&host_cycles, d_cycles, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        total_cycles += static_cast<double>(host_cycles);
    }

    double avg_cycles_per_iter = total_cycles / (trials * iterations);
    double ns_per_cycle = 1e6 / clock_rate_khz;
    double avg_ns = avg_cycles_per_iter * ns_per_cycle;
    return {avg_cycles_per_iter, avg_ns};
}

int main(int argc, char **argv) {
    size_t num_nodes = 1 << 20;     // 1M nodes ~4 MB
    size_t iterations = 1 << 22;    // number of pointer dereferences per trial
    int trials = 10;
    int compute_device = 0;
    int remote_device = 1;

    if (argc >= 2) {
        num_nodes = std::stoul(argv[1]);
    }
    if (argc >= 3) {
        iterations = std::stoul(argv[2]);
    }
    if (argc >= 4) {
        trials = std::stoi(argv[3]);
    }
    if (argc >= 5) {
        compute_device = std::stoi(argv[4]);
    }
    if (argc >= 6) {
        remote_device = std::stoi(argv[5]);
    }

    if (num_nodes < 2 || iterations == 0 || trials <= 0) {
        std::fprintf(stderr, "Invalid arguments.\n");
        return EXIT_FAILURE;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (compute_device >= device_count || remote_device >= device_count) {
        std::fprintf(stderr, "Device index out of range. Found %d GPUs.\n", device_count);
        return EXIT_FAILURE;
    }
    if (compute_device == remote_device) {
        std::fprintf(stderr, "compute_device and remote_device must differ.\n");
        return EXIT_FAILURE;
    }

    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, compute_device, remote_device));
    if (!can_access) {
        std::fprintf(stderr, "Device %d cannot access peer %d.\n", compute_device, remote_device);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(compute_device));
    cudaError_t peer_status = cudaDeviceEnablePeerAccess(remote_device, 0);
    if (peer_status != cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_CHECK(peer_status);
    } else {
        cudaGetLastError();
    }

    CUDA_CHECK(cudaSetDevice(remote_device));
    peer_status = cudaDeviceEnablePeerAccess(compute_device, 0);
    if (peer_status != cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_CHECK(peer_status);
    } else {
        cudaGetLastError();
    }

    ChainData chain_data = build_pointer_chain(num_nodes);
    const std::vector<uint32_t> &chain = chain_data.chain;
    uint32_t start_index = chain_data.start_index;

    uint32_t *local_chain = upload_chain_to_device(compute_device, chain);
    uint32_t *remote_chain = upload_chain_to_device(remote_device, chain);

    CUDA_CHECK(cudaSetDevice(compute_device));
    uint32_t *d_sink = nullptr;
    unsigned long long *d_cycles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, compute_device));
    double clock_rate_khz = static_cast<double>(prop.clockRate);

    LatencyResult local = measure_latency(compute_device, local_chain, num_nodes,
                                          iterations, trials, start_index,
                                          d_sink, d_cycles, clock_rate_khz);

    LatencyResult remote = measure_latency(compute_device, remote_chain, num_nodes,
                                           iterations, trials, start_index,
                                           d_sink, d_cycles, clock_rate_khz);

    std::printf("Pointer chain size: %zu nodes (%.2f MB)\n",
                num_nodes, num_nodes * sizeof(uint32_t) / 1e6);
    std::printf("Iterations per trial: %zu, trials: %d\n", iterations, trials);
    std::printf("Compute device: %d, Remote device: %d\n", compute_device, remote_device);
    std::printf("Clock rate: %.2f MHz\n", clock_rate_khz / 1000.0);
    std::printf("\n");
    std::printf("Local HBM latency : %.2f cycles  (~%.2f ns)\n",
                local.avg_cycles_per_iter, local.avg_ns);
    std::printf("Remote HBM latency: %.2f cycles  (~%.2f ns)\n",
                remote.avg_cycles_per_iter, remote.avg_ns);

    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_sink));

    CUDA_CHECK(cudaSetDevice(remote_device));
    CUDA_CHECK(cudaFree(remote_chain));
    CUDA_CHECK(cudaDeviceReset());

    CUDA_CHECK(cudaSetDevice(compute_device));
    CUDA_CHECK(cudaFree(local_chain));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

