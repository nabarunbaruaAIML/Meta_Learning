conda create --prefix ./env python=3.7 -y
source activate ./env
pip install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
conda env export > conda.yaml