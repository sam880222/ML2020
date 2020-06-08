import cv2
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
from model import FeatureExtractor, LabelPredictor

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)

data_dir = sys.argv[1]
model_dir = '.' #'/tmp2/b06902069/hw12'

source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # 重新將np.array 轉回 skimage.Image
    transforms.ToPILImage(),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15, fill=0),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15, fill=0),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])

source_dataset = ImageFolder(os.path.join(data_dir, 'train_data'), transform=source_transform)
target_dataset = ImageFolder(os.path.join(data_dir, 'test_data'), transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
label_predictor.load_state_dict(torch.load(os.path.join(model_dir, 'predictor_model.bin')))
feature_extractor.load_state_dict(torch.load(os.path.join(model_dir, 'extractor_model.bin')))
class_criterion = nn.CrossEntropyLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())

def train_epoch(dataloader):
    running_F_loss = 0.0
    total_hit, total_num = 0.0, 0.0
    
    for i, (data, label) in enumerate(dataloader):

        data = data.cuda()
        label = label.cuda()

        feature = feature_extractor(data)
        class_logits = label_predictor(feature)

        loss = class_criterion(class_logits, label)
        running_F_loss+= loss.item()

        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == label).item()
        total_num += data.shape[0]
        print(i, end='\r')

    return running_F_loss / (i+1), total_hit / total_num

pred = []
prob = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))
    softmax = torch.nn.Softmax(dim = 1)
    class_logits = softmax(class_logits)
    # print(torch.sum(class_logits, dim = 1))
    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    y = torch.max(class_logits, dim = 1)[0].data.cpu().detach().numpy()
    pred.append(x)
    prob.append(y)

pred = np.concatenate(pred)
prob = np.concatenate(prob)

semi_x = []
semi_y = []
for i in range(target_dataset.__len__()):
    if prob[i] >= 0.9:
        semi_x.append(target_dataset[i][0].numpy())
        semi_y.append(pred[i])

for a in source_dataset:
    semi_x.append(a[0].numpy())
    semi_y.append(a[1])


semi_x = np.array(semi_x)
semi_x = torch.from_numpy(semi_x)
semi_y = np.array(semi_y)
semi_y = torch.from_numpy(semi_y)
print(semi_x.shape)
semi_dataset = TensorDataset(semi_x, semi_y)
semi_dataloader = DataLoader(semi_dataset, batch_size=32, shuffle=True)
# semi-supervised
min_loss = np.Infinity
for epoch in range(30):
    train_F_loss, train_acc = train_epoch(semi_dataloader)

    if(train_F_loss <= min_loss):
        min_loss = train_F_loss
        torch.save(feature_extractor.state_dict(), os.path.join(model_dir, 'extractor_model_semi.bin'))
        torch.save(label_predictor.state_dict(), os.path.join(model_dir, 'predictor_model_semi.bin'))

    print('epoch {:>3d}: train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_F_loss, train_acc))
