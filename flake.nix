{
  description = "my project description";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        # let pkgs = nixpkgs.legacyPackages.${system}; in
        let pkgs = import nixpkgs { config.allowUnfree = true; system = system; }; in
        {
          devShells.default = import ./shell.nix { inherit pkgs; };
        }
      );
}
