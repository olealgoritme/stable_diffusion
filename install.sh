#!/bin/bash
pip install -r requirements.txt
pip uninstall torch -y
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install cuda-python
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
