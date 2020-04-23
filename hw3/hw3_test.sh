#!/bin/bash
wget -O model1.pth https://github.com/sam880222/ML2020/releases/download/ml/model1.pth
wget -O model2.pth https://github.com/sam880222/ML2020/releases/download/ml/model2.pth
wget -O model3.pth https://github.com/sam880222/ML2020/releases/download/ml/model3.pth
wget -O model4.pth https://github.com/sam880222/ML2020/releases/download/ml/model4.pth
wget -O model5.pth https://github.com/sam880222/ML2020/releases/download/ml/model5.pth
python3 test1.py $1
python3 test2.py $1
python3 test3.py $1
python3 test4.py $1
python3 test5.py $1
python3 polling.py $2