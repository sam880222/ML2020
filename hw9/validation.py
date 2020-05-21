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
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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

    # Second Dimesnion Reduction
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

def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    # 因為是 binary unsupervised clustering，因此取 max(acc, 1-acc)
    return max(acc, 1-acc)

# directories
checkpoint_dir = sys.argv[1]

# fix random
same_seeds(0)

# load model
model = AE().cuda()
model.load_state_dict(torch.load(checkpoint_dir))
model.eval()

# 準備 data
valX = np.load('/tmp2/b06902069/valX.npy')
valY = np.load('/tmp2/b06902069/valY.npy')

# 預測答案
latents = inference(X=valX, model=model)
pred, X_embedded = predict(latents)
acc_latent = cal_acc(valY, pred)
print('The clustering accuracy is:', acc_latent)
# 將預測結果存檔，上傳 kaggle
# save_prediction(pred, pre_path)

# 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
# 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
# save_prediction(invert(pred), pre_path)