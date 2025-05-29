# Introduce
This is a simple model to identify animal pictures, supporting training, testing and prediction
> This project build with Python 3.12.7

# Install python3
Install python first, visit:https://devops.aliyun.com/lingma/login?port=37510&state=2-5e560342dac84c018d93acd917418183

# Install torch
pip3 install torch torchvision torchaudio transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers[torch]

# Just run it
python3 main.py train
python3 main.py test
python3 main.py predict --dir=.\data\predict

