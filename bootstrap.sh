#!/usr/bin/env bash

venv_base="venv"
if [[ ! -d "$venv_base" ]]; then
    echo "Making a virtual environment at $venv_base"
    python3 -m venv "$venv_base"
fi

echo "Activating virtual environment at {$venv_base}"
source "${venv_base}/bin/activate"

echo "Installing requirements into ${venv_base}"
pip install -r requirements.txt

if [[ $(python -c "import torch; print(torch.cuda.is_available())") == "False" ]]; then
    echo "If you want to use CUDA/ROCm, uninstall PyTorch and follow the PyTorch manual install"
    echo "https://pytorch.org/get-started/locally/"
    echo "If you're on macOS, you can ignore this because macOS uses mps"
fi


echo -e "Done bootstrapping!"
echo "You can get models at https://github.com/pressjson/boredealis/releases"
