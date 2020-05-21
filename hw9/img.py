import numpy as np
import imageio
import os
imgs = np.load('/tmp2/b06902069/trainX.npy')
os.chdir('img')
for i, img in enumerate(imgs):
    imageio.imwrite('%04d.png' % i, img)