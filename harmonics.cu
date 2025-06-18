#include <cmath>
#include <chrono>
#include <cstdio>
#include <random>

using std::size_t;

void generate_random_array(double* arr, size_t size) {
    std::random_device rd;
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    
    for(size_t i = 0; i < size; i++) {
        arr[i] = dist(rd);
    }
}

void generate_array(double* arr, size_t size) {
    //std::random_device rd;
    //std::uniform_real_distribution<> dist(-1.0, 1.0);
    
    for(size_t i = 0; i < size; i++) {
        //arr[i] = dist(rd);
        arr[i] = (double) i;
    }
}

void test_serial(const double* a, const double* b, double* c, size_t size) {
    for(size_t i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void test_kernel(const double* a, const double* b, double* c, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

__host__ void test(const double* a, const double* b, double* c, size_t size) {
    size_t threads_per_block = 8*32;
    size_t nBlocks = (size + threads_per_block - 1) / threads_per_block;
    test_kernel<<<nBlocks, threads_per_block>>>(a, b, c, size);
}



int main(int argc, char* argv[]) {
    size_t n = std::stoul(argv[1]);
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_ms; // Actually seconds?

    // arrays
    size_t N = pow(2, n);
    size_t maxIndex = N-1;
    double* a = new double[N];
    double* b = new double[N];
    double* c = new double[N];

    start = high_resolution_clock::now();
    //generate_random_array(a, N);
    //generate_random_array(b, N);
    generate_array(a, N);
    generate_array(b, N);
    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    printf("Time generating arrays:%f\n", duration_ms.count());

    // Run normal version
    start = high_resolution_clock::now();

    // CODE HERE
    test_serial(a, b, c, N);
    printf("a[%lu] = %f\n", maxIndex, a[maxIndex]);
    printf("b[%lu] = %f\n", maxIndex, b[maxIndex]);
    printf("c[%lu] = %f\n", maxIndex, c[maxIndex]);

    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    printf("non-CUDA code time: %f\n", duration_ms.count());

    // Run CUDA version
    double* dA;
    double* dB;
    double* dC;
    cudaMalloc(&dA, N*sizeof(double));
    cudaMalloc(&dB, N*sizeof(double));
    cudaMalloc(&dC, N*sizeof(double));
    cudaMemcpy(dA, a, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, N*sizeof(double), cudaMemcpyHostToDevice);
    start = high_resolution_clock::now();

    c[maxIndex] = 0;
    printf("c[%lu] = %f\n", maxIndex, c[maxIndex]);
    test(dA, dB, dC, N);
    cudaDeviceSynchronize();

    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cudaMemcpy(c, dC, N*sizeof(double), cudaMemcpyDeviceToHost);
    printf("c[%lu] = %f\n", maxIndex, c[maxIndex]);
    printf("CUDA code time: %f\n", duration_ms.count());

    // free memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}