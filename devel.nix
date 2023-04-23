let
nixpkgs = import <nixpkgs> {};

# LAPACK library
lapack = nixpkgs.lapack;

# CUDA library
cuda = nixpkgs.cudaPackages.cudatoolkit;

# CMake build system
cmake = nixpkgs.cmake;
cublas = nixpkgs.cudaPackages.libcublas;
blas = nixpkgs.blas;
in
# Create a Nix derivation for the development environment
with nixpkgs;

stdenv.mkDerivation {
    name = "my-cuda-dev-env";
    buildInputs = [
# Add LAPACK and CUDA libraries as build dependencies
        lapack
            cuda
            cmake
            cublas
            blas
    ];
    shellHook = ''
# Set the necessary environment variables for CUDA
        export PATH=$PATH:${cuda}/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${cuda}/lib64
        export PATH=$PATH:${cublas}/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${cublas}/lib64
        export PATH=$PATH:${blas}/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${blas}/lib64

# Set the necessary environment variables for LAPACK
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lapack}/lib64
        '';
}
