import os
import json
import shutil
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import re
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import models
from torch.optim import lr_scheduler

BATCH_SIZE = 256
IMAGE_SIZE = 64
CROP_SIZE = 52
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 데이터 경로
#data_dir = '/Users/bamgee/Downloads/071.시설 작물 질병 진단/01.데이터'
data_dir = '/Volumes/Samsung_T5/data/071.시설 작물 질병 진단/01.데이터'
# train_images_dir = os.path.join(data_dir, '1.Training/원천데이터/07.애호박')
# train_labels_dir = os.path.join(data_dir, '1.Training/라벨링데이터/07.애호박')
# val_images_dir = os.path.join(data_dir, '2.Validation/원천데이터/07.애호박')
# val_labels_dir = os.path.join(data_dir, '2.Validation/라벨링데이터/07.애호박')
train_images_dir = os.path.join(data_dir, '1.Training/원천데이터/10.참외')
train_labels_dir = os.path.join(data_dir, '1.Training/라벨링데이터/10.참외')
val_images_dir = os.path.join(data_dir, '2.Validation/원천데이터/10.참외')
val_labels_dir = os.path.join(data_dir, '2.Validation/라벨링데이터/10.참외')

# JSON 라벨 파일 로드 함수
def load_json_labels(labels_dir):
    labels_data = {}
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.json'):
            with open(os.path.join(labels_dir, label_file), 'r') as file:
                data = json.load(file)
                image_name = data["description"]["image"]
                annotations = data["annotations"]
                labels_data[image_name] = annotations
    return labels_data

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, img_dir, labels_dir, transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_labels = self.create_img_label_mapping()
        # self.printed_labels = set()

    def create_img_label_mapping(self):
        mapping = {}
        for label_subdir in os.listdir(self.labels_dir):
            label_path = os.path.join(self.labels_dir, label_subdir)
            if not os.path.isdir(label_path):
                continue

            # 정규 표현식을 사용하여 '[라벨]' 접두사 제거
            img_subdir = re.sub(r'\[.*?\]', '', label_subdir).strip()
            img_path = os.path.join(self.img_dir, img_subdir)

            # 디버깅 정보
            print(f"원본 라벨 디렉토리 이름: {label_subdir}")
            print(f"변환된 이미지 디렉토리 이름: {img_subdir}")

            if not os.path.isdir(img_path):
                print(f"경로가 존재하지 않습니다: {img_path}")
                continue

            for label_file in os.listdir(label_path):
                if label_file.endswith('.json'):
                    with open(os.path.join(label_path, label_file), 'r') as file:
                        data = json.load(file)
                        image_name = data["description"]["image"]
                        annotations = data["annotations"]

                        # 원천 데이터 이미지 경로와 라벨 매핑
                        mapping[os.path.join(img_path, image_name)] = annotations

        return mapping

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = list(self.img_labels.items())[idx]
        image = Image.open(img_path)
        #label = label['disease']  # 'disease' 값을 사용

        # if label not in self.printed_labels:
        #     print(f"라벨: {label}")
        #     self.printed_labels.add(label)

        # TODO : 작물에 맞게 수정
        if label['disease'] == 16:  # 참외노균병
            new_label = 1
        elif label['disease'] == 17:    #참외흰가루병
            new_label = 2
        else:
            new_label = 0   # 정상

        if self.transform:
            image = self.transform(image)

        return image, new_label

# 데이터 변환 설정 (전이 학습을 위해)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    'val': transforms.Compose([
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.RandomCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
}

# # 데이터셋 및 트랜스폼 설정
# transform_base = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
# # 사용자 정의 데이터셋 생성
# train_dataset = CustomDataset(train_images_dir, train_labels_dir, transform=transform_base)
# val_dataset = CustomDataset(val_images_dir, val_labels_dir, transform=transform_base)

# 라벨 데이터 로드
train_labels = load_json_labels(train_labels_dir)
val_labels = load_json_labels(val_labels_dir)

# 사용자 정의 데이터셋 생성
train_dataset = CustomDataset(train_images_dir, train_labels_dir, transform=data_transforms['train'])
val_dataset = CustomDataset(val_images_dir, val_labels_dir, transform=data_transforms['val'])

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 전이 학습 모델 설정 및 학습
def train_resnet(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 각 에포크마다 학습 및 검증
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반복
            loader = train_loader if phase == 'train' else val_loader
            # for inputs, labels in train_loader if phase == 'train' else val_loader:
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # epoch_loss = running_loss / len(train_loader.dataset if phase == 'train' else val_loader.dataset)
            # epoch_acc = running_corrects.double() / len(train_loader.dataset if phase == 'train' else val_loader.dataset)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# 전이 학습 모델 로드 및 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# TODO : 작물별 가중치
# 클래스 가중치 설정
# class_weights = torch.tensor([1.0, 12.0, 12.0], dtype=torch.float).to(DEVICE)  # '정상', '질병1', '질병2'의 가중치

## 가지
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 고추
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 토마토
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 수박
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 참외
class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 딸기
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 상추
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 가지
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 단호박
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 쥬키니호박
# class_weights = torch.tensor([1.0, 10.0], dtype=torch.float).to(DEVICE)

## 오이
# class_weights = torch.tensor([1.0, 10.0], dtype=torch.float).to(DEVICE)

## 애호박
# class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float).to(DEVICE)

## 포도
# class_weights = torch.tensor([1.0, 10.0], dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()

resnet = models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

# 모델의 `fc` 레이어를 교체
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 3)
resnet = resnet.to(DEVICE)

optimizer_ft = optim.Adam(resnet.fc.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 전이 학습 모델 학습
EPOCH = 30
model_resnet50 = train_resnet(resnet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCH)

# TODO : 작물에 따라 이름 수정
# 모델 저장
torch.save(model_resnet50.state_dict(), '참외_resnet50.pth')