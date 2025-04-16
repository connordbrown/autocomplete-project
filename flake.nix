{
  description = "Python Packages for Data Science and NLP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3.override {
          self = python;
          packageOverrides = python-self: python-super: {
            # Potentially problematic packages - override checks if needed
            redshift-connector = python-super.redshift-connector.overridePythonAttrs (oldAttrs: { doCheck = false; });
            google-cloud-bigquery = python-super.google-cloud-bigquery.overridePythonAttrs (oldAttrs: { meta = oldAttrs.meta // { broken = false; }; doCheck = false; propagatedBuildInputs = oldAttrs.propagatedBuildInputs ++ [ python-super.packaging ]; });
          };
        };
      in
      with pkgs;
      {
        devShells.default = mkShell {
          name = "data-science-nlp-dev";
          packages = [
            # Non-Python dependencies if any (e.g., build tools)
            (python.withPackages (p: with p; [
              pandas
              pip
              scikit-learn
              nltk
              numpy
              torch
              transformers
              datasets
              pyarrow
              rouge-score
              spacy
            ]))
          ];
          shellHook = ''
            echo "Development environment is ready!"
          '';
        };
      });
}