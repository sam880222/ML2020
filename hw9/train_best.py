import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import random
import torch
import torch.nn as nn
from torch import optim
import os
import torchvision.transforms as transforms
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

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
    # image_list = np.transpose(image_list, (0, 3, 1, 2)) # (-1, 32, 32, 3) to (-1, 3, 32, 32)
    # image_list = (image_list / 255.0) * 2 - 1   # Normalize to (-1, 1)
    # image_list = image_list.astype(np.float32)
    return image_list

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            # transforms.RandomCrop(32, pad_if_needed=True, padding_mode='symmetric'),
                            transforms.RandomHorizontalFlip(),
                            # transforms.RandomVerticalFlip(),
                            # transforms.RandomRotation(15),
                            transforms.ToTensor()
                        ])
        images = self.transform(images)
        # if idx == 0:
        #     print(images)
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
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = self.fc(x1.view(x1.size()[0], -1))
        x = self.defc(x1)
        x  = self.decoder(x.view(x1.size()[0], 512, 2, 2))
        # x = self.decoder(x1)
        return x1, x

# directories
train_dir = sys.argv[1]
checkpoint_dir = sys.argv[2]

# load data
trainX = np.load(train_dir)
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

# fix random
same_seeds(0)

# load model
model = AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

model.train()
n_epoch = 200

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

min_loss = np.Infinity
# 主要的訓練過程
for epoch in range(n_epoch):
    total_num = 0
    total_loss = 0
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if (epoch+1) % 10 == 0:
        #     torch.save(model.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch+1))
        total_num += len(data)
        total_loss += loss.item() * len(data)
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, total_loss/total_num))
    if total_loss < min_loss:
        total_loss = min_loss
# 儲存 model
torch.save(model.state_dict(), checkpoint_dir)
