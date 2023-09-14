# RWKV.f90

RWKV.f90 is a port of the original [RWKV-LM](https://github.com/BlinkDL/RWKV-LM), an open-source large language model initially developed in Python, into Fortran. Renowned for its robust capabilities in scientific and engineering computations, the primary focus of this project is to explore the potential of Fortran within the realm of Artificial Intelligence.

Please note that this is an ongoing project and we welcome contributions.

# Prerequisites
Before you start, ensure that you have the following installed on your system:

- [cmake](https://cmake.org/download/)
- [gfortran](https://gcc.gnu.org/wiki/GFortranBinaries)
- [go](https://golang.org/dl/)

# Step-by-Step Guide

This section provides a detailed guide to setting up and running the RWKV.f90 project. The process includes downloading a model, moving the model to the correct directory, converting the model, building the project, and running it.

## 1. Downloading the Model

Download the `rwkv-4-world` model of your choice from the following options available at [huggingface.co](https://huggingface.co/BlinkDL/rwkv-4-world/tree/main):

- [0.1B](https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-0.1B-v1-20230520-ctx4096.pth)
- [0.4B](https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-0.4B-v1-20230529-ctx4096.pth)
- [1.5B](https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth)
- [3B](https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-3B-v1-20230619-ctx4096.pth) - **Recommended**
- [7B](https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-7B-v1-20230626-ctx4096.pth)
  
Once you have downloaded your chosen model, move it to the 'models' directory in the project's root folder.

## 2. Model Conversion 

After the desired model is in the correct location, convert it using the model converter. Make sure to adjust the filename according to the model you have downloaded. Run the following commands in your terminal:

```bash
cd go/model_converter
go run model_converter.go ../../models/<YOUR-MODEL-NAME>.pth
```

Replace `<YOUR-MODEL-NAME>` with the actual name of the downloaded model file.

## 3. Build the Project

With the model conversion done, you can now build the project. 

Navigate to the root directory of the project and run the following commands:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

After successful compilation, the build will generate `rwkv-cli`, which is a stand-alone executable to test the language model on the terminal. Additionally, it will produce `librwkv.a`, a library that can be linked into other languages.

### 3.1 Choosing a BLAS Implementation

The build system uses CMake's `FindBLAS` to automatically search for a suitable BLAS (Basic Linear Algebra Subprograms) library for matrix multiplication operations (`matmul`). This approach gives you flexibility in selecting the BLAS library you'd like to use.

#### Guiding the Selection with `BLA_VENDOR`

You can guide the BLAS library selection process by setting the `BLA_VENDOR` variable when running `cmake`. Here are some options:

- `BLA_VENDOR=OpenBLAS`: This utilizes the OpenBLAS library, an open-source implementation of the BLAS API.
- `BLA_VENDOR=Apple`: This leverages the Apple Accelerate Framework, which is optimized for Apple hardware.
- Leave `BLA_VENDOR` unset: This allows CMake to choose the best available option, which could default to using the intrinsic `matmul` function in Fortran if no external BLAS libraries are found.

Here's an example that uses OpenBLAS:

```bash
BLA_VENDOR=OpenBLAS cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

#### Special Instructions for OpenBLAS on macOS with Homebrew

On macOS, the Homebrew package manager doesn't symlink BLAS libraries to `/usr/local` in order to avoid conflicts with Apple's native Accelerate framework. 

Therefore, to use OpenBLAS installed via Homebrew you also need to specify the `CMAKE_PREFIX_PATH` when running `cmake`.

For example, you would run:

```bash
BLA_VENDOR=OpenBLAS cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/openblas/lib" .. && make
```

#### Special Instructions for Intel MKL implementation of BLAS

Please refer to the following documentation of CMake FindBLAS:
https://cmake.org/cmake/help/latest/module/FindBLAS.html#intel-mkl

For example, after installing Intel oneAPI MKL under `/opt/intel/oneapi`, the
project can be built with a sequence of commands like this:

```bash
mkdir build && cd build
. /opt/intel/oneapi/setvars.sh
cmake -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Intel10_64lp ..
make
```

## 4. Run the Project

After successfully building the project, you're ready to run it. 

Navigate to the build directory and run the `rwkv-cli` executable with the following command:

```bash
OMP_NUM_THREADS=8 ./rwkv-cli -tokenizer ../models/rwkv_vocab_v20230424.csv -model ../models/<YOUR-CONVERTED-MODEL-NAME> 2> >(while read line; do echo -e "\e[01;31m$line\e[0m" >&2; done) 
```

Replace `<YOUR-CONVERTED-MODEL-NAME>` with the actual name of your converted model file.

## 4.1 Run with Speculative Sampling

[Speculative sampling](https://arxiv.org/pdf/2302.01318.pdf) can double the speed of the generation. However, the current implementation is experimental with known TODOs.

To leverage this, pair your main model with a faster draft model. Consider using the `0.1B` model as your draft, which has been observed to be surprisingly robust for this purpose.

```bash
OMP_NUM_THREADS=8 ./rwkv-cli -tokenizer ../models/rwkv_vocab_v20230424.csv -model ../models/<YOUR-CONVERTED-MODEL-NAME> -draft ../models/<YOUR-CONVERTED-SMALLER-MODEL-NAME> 2> >(while read line; do echo -e "\e[01;31m$line\e[0m" >&2; done) 
```

Replace placeholders with appropriate model names.

# References

The paper provides more details about the RWKV concept and its applications. It's recommended to read this paper for a better understanding of the project and its scientific background.

```console
@misc{peng2023rwkv,
      title={RWKV: Reinventing RNNs for the Transformer Era}, 
      author={Bo Peng and Eric Alcaide and Quentin Anthony and Alon Albalak and Samuel Arcadinho and Huanqi Cao and Xin Cheng and Michael Chung and Matteo Grella and Kranthi Kiran GV and Xuzheng He and Haowen Hou and Przemyslaw Kazienko and Jan Kocon and Jiaming Kong and Bartlomiej Koptyra and Hayden Lau and Krishna Sri Ipsit Mantri and Ferdinand Mom and Atsushi Saito and Xiangru Tang and Bolun Wang and Johan S. Wind and Stansilaw Wozniak and Ruichong Zhang and Zhenyuan Zhang and Qihang Zhao and Peng Zhou and Jian Zhu and Rui-Jie Zhu},
      year={2023},
      eprint={2305.13048},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
