{
  description = "MPMC Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { 
        inherit system;
        config.allowUnfree = true;
      };

    in
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
            lapack
            cudaPackages.cudatoolkit
            cmake
            cudaPackages.libcublas
            gcc
            blas
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
      };
    });
}
