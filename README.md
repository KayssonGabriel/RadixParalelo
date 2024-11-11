# Radix Sort paralelo em CUDA

Benchmark de Radix Sort em GPU. [main.cu](main.cu) gera vetores e usa a ordenação estável de inteiros da biblioteca CUB, incluída no CUDA Toolkit. O tempo inclui as alocações, a ordenação e as cópias entre CPU e GPU.

## Como compilar

Use CMake 3.29 ou superior, CUDA Toolkit com suporte a CUDA C++20, um compilador de host compatível e uma GPU NVIDIA com driver compatível.

~~~sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
~~~

O alvo definido em [CMakeLists.txt](CMakeLists.txt) é `RadixSortCUDA`. No Linux, execute `build/RadixSortCUDA`. Com o gerador do Visual Studio no Windows, execute `build/Release/RadixSortCUDA.exe`.

O programa confere se a saída está em ordem antes de exibir o tempo. Os cinco tamanhos vão de 100 a 10 milhões de elementos. A versão para CPU está em [RadixSerial](https://github.com/KayssonGabriel/RadixSerial).
