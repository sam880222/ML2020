#!/bin/bash
if [ -f "./w2v_all.model" ]; then
    echo "w2v_all.model exist"
else
    wget -O w2v_all.model https://github.com/sam880222/ML2020/releases/download/ml/w2v_all.model
fi
if [ -f "./model.pth" ]; then
    echo "model.pth exist"
else
    wget -O model.pth https://github.com/sam880222/ML2020/releases/download/ml/hw4_model.pth
fi

python3 test.py $1 $2