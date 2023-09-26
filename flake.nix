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
            clang-tools
            lapack
            cudaPackages.cudatoolkit
            cmake
            cudaPackages.libcublas
            gcc
            blas
            valgrind

            python3
            python311Packages.python-lsp-server
            python311Packages.autopep8
            python310Packages.ujson
            python310Packages.numpy
            python310Packages.scipy
            python310Packages.pluggy
            python310Packages.docstring-to-markdown
            python310Packages.jedi
        ];
      };
    });
}
