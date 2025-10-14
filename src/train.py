import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

# Dataset 정의
class SongSequenceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.song_folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, idx):
        song_folder = self.song_folders[idx]
        # 파일 이름 순서대로 정렬하여 0~3(인트로), 4~7(버스), 8~11(코러스) 순서를 보장
        image_paths = sorted(glob.glob(os.path.join(song_folder, '*.png')))
        
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        with open(os.path.join(song_folder, 'labels.txt'), 'r') as f:
            labels = [int(l.strip()) for l in f.read().split(',')]
        
        return torch.stack(images), torch.LongTensor(labels)

# CRNN 모델 정의
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )
        # 128x128 이미지 기준 CNN 출력 크기: 128 * 8 * 8 = 8192
        self.rnn = nn.LSTM(input_size=8192, hidden_size=256, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, seq_len, -1)
        r_out, _ = self.rnn(r_in)
        out = self.classifier(r_out)
        return out

# 학습 스크립트
if __name__ == '__main__':
    # 하이퍼파라미터
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 4
    EPOCHS = 50
    NUM_CLASSES = 3 # intro, verse, chorus
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 데이터 로더 (모든 시퀀스 길이가 12로 동일하므로 collate_fn이 더 이상 필요 없음)
    train_dataset = SongSequenceDataset(data_dir='dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 모델, 손실함수, 옵티마이저
    model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(sequences)
            
            loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'crnn_feature_extractor.pth')
    print("모델 저장 완료")