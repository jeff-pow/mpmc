{ pkgs ? import <nixpkgs> {} }:
with pkgs;

stdenv.mkDerivation {
    name = "MPMC Shell";
    buildInputs = [
        pkgs.lapack
        pkgs.cudaPackages.cudatoolkit
        pkgs.cmake
        pkgs.cudaPackages.libcublas
        pkgs.blas
    ];
    shellHook = ''
        export PATH=$PATH:${pkgs.cudaPackages.cudatoolkit}/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.cudaPackages.cudatoolkit}/lib64
        export PATH=$PATH:${pkgs.cudaPackages.libcublas}/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.cudaPackages.libcublas}/lib64
        export PATH=$PATH:${pkgs.blas}/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.blas}/lib64
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lapack}/lib64
        '';
}
