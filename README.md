Deep Learning TF Playground
===========================

Setup (Apple Silicon, arm64)
----------------------------
- Use a native (non-Rosetta) shell: uname -m should print arm64. If it prints x86_64, start an arm shell with arch -arm64 zsh or turn off Open using Rosetta for Terminal.
- Create/activate the env (Miniforge/conda example):
  - conda create -n dl-tf-arm python=3.11
  - conda activate dl-tf-arm
- Install deps and this repo in editable mode (repo root):
  - pip install --upgrade pip
  - pip install tensorflow-macos tensorflow-metal matplotlib
  - pip install -e .

Using the notebook
------------------
- Open the notebook in VS Code and pick the dl-tf-arm interpreter (~/miniforge3/envs/dl-tf-arm/bin/python).
- Run cells from the top. Imports like from src.data import get_mnist_datasets work after pip install -e . with the current flat layout.

Troubleshooting
---------------
- ModuleNotFoundError: src: ensure the kernel is dl-tf-arm and you ran pip install -e . in that env; restart the kernel after installing.
- platform.machine() shows x86_64: you are under Rosetta—switch to an arm64 shell and recreate/reuse the env.
- MNIST download writes to ~/.keras/datasets; set KERAS_HOME to a writable path if needed.
- Missing matplotlib: pip install matplotlib in the active env.

