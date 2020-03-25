import numpy as np
import sys
with open("train.csv") as f:
    x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = str)    
x = x.T
single_colume = ['age', 'wage per hour', 'capital gains', 'capital losses', 'dividends from stocks', 'weeks worked in year', 'num persons worked for employer']
feature_to_remove = ['detailed household and family stat', 'detailed industry recode', 'reason for unemployment', 'country of birth father', \
    'detailed occupation recode', 'country of birth mother']#['family members under 18', 'marital stat', 'major industry code', 'major occupation code', 'hispanic origin', 'region of previous residence']#[, 'country of birth father', 'country of birth mother', 'sex', 'live in this house 1 year ago']
data = []
for i in range(len(x)):
    if x[i, 0] in feature_to_remove:
        continue
    if x[i, 0] in single_colume:
        data.append([x[i, 0], 1])
    elif x[i, 0] == 'detailed household and family stat':
        data.append([x[i, 0], len(list(set(x[i, 1:]))) + 2])
    else:
        data.append([x[i, 0], len(list(set(x[i, 1:])))])
data = data[0:-1]
index = dict()
place = 0
for d in data:
    index[d[0]] = [place]
    place += d[1]
dim = place
#print(dim)
print(index)