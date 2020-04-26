#!/bin/bash
if [ -f "./w2v_all.model" ]; then
    echo "w2v_all.model exist"
else
    wget -O w2v_all.model https://github.com/sam880222/ML2020/releases/download/ml/w2v_all.model
fi
python3 train.py $1 $2