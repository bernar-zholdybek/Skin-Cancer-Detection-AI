{ pkgs, lib, config, inputs, ... }:

{
  env.GREET = "devenv";

  packages = [
    pkgs.git
    pkgs.jupyter
    pkgs.python313Packages.tensorflowWithCuda
    pkgs.python313Packages.keras
    pkgs.python313Packages.matplotlib
    pkgs.python313Packages.customtkinter
    pkgs.python313Packages.pillow
    pkgs.python313Packages.pandas
    pkgs.python313Packages.seaborn
  ];
  
}
