import sys
import pandas as pd
import numpy as np
from google.colab import drive 

# import training data
data = pd.read_csv('./train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
print(type(data))
data[data == 'NR'] = 0
raw_data = data.to_numpy()
print(raw_data)
