#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

void conferirCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void radixSortCuda(std::vector<int>& valores) {
    if (valores.size() < 2) {
        return;
    }
    if (valores.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("Vetor grande demais para esta configuração.");
    }

    const auto tamanho = static_cast<int>(valores.size());
    thrust::device_vector<int> entrada(valores.begin(), valores.end());
    thrust::device_vector<int> saida(tamanho);
    auto* dadosEntrada = thrust::raw_pointer_cast(entrada.data());
    auto* dadosSaida = thrust::raw_pointer_cast(saida.data());

    // CUB informa primeiro o espaço temporário necessário à ordenação.
    std::size_t bytesTemporarios = 0;
    conferirCuda(cub::DeviceRadixSort::SortKeys(
        nullptr, bytesTemporarios, dadosEntrada, dadosSaida, tamanho));
    thrust::device_vector<unsigned char> temporario(bytesTemporarios);
    conferirCuda(cub::DeviceRadixSort::SortKeys(
        thrust::raw_pointer_cast(temporario.data()), bytesTemporarios,
        dadosEntrada, dadosSaida, tamanho));
    conferirCuda(cudaDeviceSynchronize());
    thrust::copy(saida.begin(), saida.end(), valores.begin());
}

int main() {
    try {
        const std::vector<int> tamanhos{100, 1000, 10000, 1000000, 10000000};
        for (const int tamanho : tamanhos) {
            std::vector<int> valores(tamanho);
            std::generate(valores.begin(), valores.end(), [] { return std::rand() % 10000000; });

            const auto inicio = std::chrono::steady_clock::now();
            radixSortCuda(valores);
            const std::chrono::duration<double> duracao = std::chrono::steady_clock::now() - inicio;

            if (!std::is_sorted(valores.begin(), valores.end())) {
                throw std::runtime_error("O resultado da GPU está fora de ordem.");
            }
            std::cout << "Tempo de execução CUDA para " << tamanho << " elementos: "
                      << std::fixed << std::setprecision(6) << duracao.count() << " segundos\n";
        }
    } catch (const std::exception& erro) {
        std::cerr << "Falha no Radix Sort CUDA: " << erro.what() << '\n';
        return 1;
    }
    return 0;
}
