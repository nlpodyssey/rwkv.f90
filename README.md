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
- [3B](https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-3B-v1-20230619-ctx4096.pth) - **Recommended** for a balance of performance and computational efficiency
  
Once you have downloaded your chosen model, move it to the 'models' directory in the project's root folder.

## 2. Model Conversion 

After the desired model is in the correct location, convert it using the model converter. Make sure to adjust the filename according to the model you have downloaded. Run the following commands in your terminal:

```console
cd model_converter
go run model_converter.go ../models/<YOUR-MODEL-NAME>.pth
```

Replace `<YOUR-MODEL-NAME>` with the actual name of the downloaded model file.

## 3. Build the Project

With the model conversion done, you can now build the project. Navigate to the root directory of the project and run the following commands:

```console
mkdir build
cd build
cmake ..
make
```

### 3.1 Choosing a BLAS Implementation

The underlying BLAS (Basic Linear Algebra Subprograms) implementation that's used for matrix multiplication operations (`matmul`) can be chosen by providing the `-DBLAS_LIBRARY` flag during the CMake configuration step.

Three options are available:

- `DBLAS_LIBRARY=OpenBLAS`: This utilizes the OpenBLAS library, which is an open-source implementation of the BLAS API. Specify in the `CMakeList.txt` the directory where OpenBLAS library is located.
- `DBLAS_LIBRARY=Accelerate`: This leverages the Apple Accelerate Framework, which is highly optimized for Apple hardware.
- `DBLAS_LIBRARY=Fortran` or leaving it unset: This defaults to using the Fortran's intrinsic matmul function, which allows the Fortran compiler to handle matrix operations without an external library.

To select an option, include the relevant flag when running `cmake`. For example, to use the OpenBLAS library, you would run:

```console
cmake -DBLAS_LIBRARY=OpenBLAS ..
```

By not specifying a `-DBLAS_LIBRARY` flag or setting it to `Fortran`, the build will rely on the intrinsic `matmul` function provided by the compiler (resulting in a slow execution).

## 4. Run the Project

Once the project is built, you can now run it. The following command also colors stderr outputs in red for better error visibility. Adjust the command as necessary, according to your downloaded and converted model name:

```console
./rwkv ../models/rwkv_vocab_v20230424.csv ../models/<YOUR-CONVERTED-MODEL-NAME> 2> >(while read line; do echo -e "\e[01;31m$line\e[0m" >&2; done) 
```

Replace `<YOUR-CONVERTED-MODEL-NAME>` with the name of your converted model file.

That's all! You have successfully set up and run the rwkv.f90 project. If you encounter any issues, please raise them in the issue tracker.

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
