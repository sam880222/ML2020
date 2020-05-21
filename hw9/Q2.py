import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torch
import torch.nn as nn
from torch import optim
import os
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess(image_list):
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2)) # (-1, 32, 32, 3) to (-1, 3, 32, 32)
    image_list = (image_list / 255.0) * 2 - 1   # Normalize to (-1, 1)
    image_list = image_list.astype(np.float32)
    return image_list

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),         
        )

        self.fc = nn.Sequential(
            nn.Linear(512*2*2, 512),
            nn.Dropout(0.5),
            nn.ReLU(True),
            # nn.Linear(512, 256),
            # nn.Dropout(0.5),
            # nn.ReLU(True),
        )

        self.defc = nn.Sequential(
            # nn.Linear(256, 1024),
            # nn.Dropout(0.5),
            # nn.ReLU(True),
            nn.Linear(512, 512*2*2),
            nn.Dropout(0.5),
            nn.ReLU(True),
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = self.fc(x1.view(x1.size()[0], -1))
        x = self.defc(x1)
        x  = self.decoder(x.view(x1.size()[0], 512, 2, 2))
        # x = self.decoder(x1)
        return x1, x

def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # First Dimension Reduction
    # transformer = KernelPCA(n_components=128, kernel='rbf', random_state = 0, n_jobs=-1)
    # kpca = transformer.fit_transform(latents)
    # print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2, random_state = 0, n_jobs = -1).fit_transform(latents)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

# directories
train_dir = '/tmp2/b06902069/trainX.npy'
checkpoint_dir = sys.argv[1]
# pre_path = sys.argv[3]

# fix random
same_seeds(0)

# load model
model = AE().cuda()
model.load_state_dict(torch.load(checkpoint_dir))
model.eval()

# 準備 data
trainX = np.load(train_dir)

# 預測答案
# latents = inference(X=trainX, model=model)
# pred, X_embedded = predict(latents)

import matplotlib.pyplot as plt

# 畫出原圖
plt.figure(figsize=(10,4))
indexes = [1,2,3,6,7,9]
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i+1, xticks=[], yticks=[])
    plt.imshow(img)
trainX_preprocessed = preprocess(trainX)
dataset = Image_Dataset(trainX_preprocessed[indexes,])
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
# 畫出 reconstruct 的圖
# inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
for x in dataloader:
    latents, recs = model(x.cuda())
# print(recs[recs<0])
recs = ((recs+1)/2 ).cpu().detach().numpy()
recs = recs.transpose(0, 2, 3, 1)
for i, img in enumerate(recs):
    plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
    plt.imshow(img)
  
plt.tight_layout()
plt.savefig('Q2.png')