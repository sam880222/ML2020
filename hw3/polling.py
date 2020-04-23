import numpy as np
import sys
pre1 = np.load("pre1.npy")
pre2 = np.load("pre2.npy")
pre3 = np.load("pre3.npy")
pre4 = np.load("pre4.npy")
pre5 = np.load("pre5.npy")

# for i in range(len(pre1)):
#     if pre2[i] == pre3[i]:
#         pre1[i] = pre2[i]

pre = np.concatenate((pre1.reshape(-1, 1), pre2.reshape(-1, 1), pre3.reshape(-1, 1), pre4.reshape(-1, 1), pre5.reshape(-1, 1)), axis=1)
for i in range(len(pre1)):
    k, counts = np.unique(pre[i], return_counts=True)
    if np.max(counts) >= 3:
        pre1[i] = k[np.argmax(counts)]
    
with open(sys.argv[1], 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(pre1):
        f.write('{},{}\n'.format(i, y))