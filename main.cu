#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// Kernel CUDA para realizar o Counting Sort paralelamente
__global__ void countingSortKernel(int* arr, int* count, int* output, int n, int exp)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n)
    {
        int digit = (arr[idx] / exp) % 10;
        atomicAdd(&count[digit], 1);
    }

    __syncthreads(); // Sincroniza os threads após o cálculo das contagens

    if (idx < 10)
    {
        for (int i = 1; i < 10; i++)
        {
            count[i] += count[i - 1];
        }
    }

    __syncthreads(); // Sincroniza antes de mover os elementos para a saída

    if (idx < n)
    {
        int digit = (arr[idx] / exp) % 10;
        int position = --count[digit];
        output[position] = arr[idx];
    }
}

// Função de Radix Sort com CUDA
void radixSortCuda(vector<int>& arr)
{
    int n = arr.size();
    int *d_arr, *d_count, *d_output;

    // Alocação de memória na GPU
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_count, 10 * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    // Copiar dados para a GPU
    cudaMemcpy(d_arr, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, 10 * sizeof(int)); // Limpar o contador de dígitos

    int maxElem = *max_element(arr.begin(), arr.end());

    // Realiza Counting Sort para cada dígito
    for (int exp = 1; maxElem / exp > 0; exp *= 10)
    {
        // Lançamento de kernel para o Counting Sort
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        countingSortKernel<<<numBlocks, blockSize>>>(d_arr, d_count, d_output, n, exp);

        // Espera o kernel terminar
        cudaDeviceSynchronize();

        // Copiar os resultados de volta para a memória do host
        cudaMemcpy(arr.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

        // Limpar o contador de dígitos para o próximo dígito
        cudaMemset(d_count, 0, 10 * sizeof(int));
    }

    // Liberação de memória na GPU
    cudaFree(d_arr);
    cudaFree(d_count);
    cudaFree(d_output);
}

int main()
{
    vector<int> sizes = {100, 1000, 10000, 1000000, 10000000};

    // Itera sobre os diferentes tamanhos de entrada
    for (int n : sizes)
    {
        vector<int> arr(n);
        // Gera números aleatórios para o array
        for (int i = 0; i < n; i++)
        {
            arr[i] = rand() % 10000000;
        }

        // Inicia a medição de tempo
        auto start = chrono::high_resolution_clock::now();
        radixSortCuda(arr);
        auto end = chrono::high_resolution_clock::now();

        // Calcula a duração e imprime o tempo de execução
        chrono::duration<double> duration = end - start;
        cout << "Tempo de execucao Paralelo (CUDA) para " << n << " elementos: "
            << fixed << setprecision(6) << duration.count() << " segundos" << endl;
    }

    return 0;
}
