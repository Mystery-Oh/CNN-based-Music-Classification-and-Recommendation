import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import glob
import random

# Transform

transform_augmented = nn.Sequential(
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.05),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
)

# 기본용 (Anchor, Negative, Validation용) - 정규화만 수행
transform_basic = nn.Sequential(
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
)


# Dataset 정의
class PreprocessedTripletDataset(Dataset):
    def __init__(self, data_dir, transform_basic, transform_augmented):
        self.data_dir = data_dir
        self.transform_basic = transform_basic
        self.transform_augmented = transform_augmented
        
        # .pt 파일 목록 스캔
        self.file_paths = glob.glob(os.path.join(data_dir, '*_intro.pt'))
        print(f"[{data_dir}] 로드됨: 총 {len(self.file_paths)}개 파일")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Anchor (기준) 로드
        anchor_path = self.file_paths[idx]
        # torch.load는 디스크 I/O가 매우 빠름
        data_tensor = torch.load(anchor_path) # [4, 3, 128, 128]
        
        # Anchor에는 기본 변환 적용
        anchor = self.transform_basic(data_tensor)
        
        # Positive (긍정) - 같은 데이터에 증강 적용
        positive = self.transform_augmented(data_tensor)
        
        # Negative (부정) - 다른 파일 로드
        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.file_paths) - 1)
        
        negative_tensor = torch.load(self.file_paths[negative_idx])
        negative = self.transform_basic(negative_tensor)
        
        return anchor, positive, negative

# 모델
class CRNN_Metric(nn.Module):
    def __init__(self, rnn_hidden_size=256, embedding_dim=512):
        super(CRNN_Metric, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(4, 4)
        )
        self.rnn = nn.LSTM(8192, rnn_hidden_size, 2, batch_first=True, bidirectional=True)
        self.embedding = nn.Linear(rnn_hidden_size * 2, embedding_dim)

    def forward(self, x):
        B, S, C, H, W = x.size()
        c_out = self.cnn(x.view(B*S, C, H, W))
        r_out, _ = self.rnn(c_out.view(B, S, -1))
        return nn.functional.normalize(self.embedding(r_out.mean(dim=1)), p=2, dim=1)

# --- 4. 실행 ---
if __name__ == '__main__':
    # 설정
    BATCH_SIZE = 64 
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 경로 (전처리된 폴더로 지정)
    TRAIN_DIR = 'dataset_preprocessed_chroma/train'
    VAL_DIR = 'dataset_preprocessed_chroma/val'
    
    # 워커 설정
    num_workers = min(8, os.cpu_count() - 2) 
    
    # 데이터셋 & 로더
    train_ds = PreprocessedTripletDataset(TRAIN_DIR, transform_basic, transform_augmented)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    val_ds = PreprocessedTripletDataset(VAL_DIR, transform_basic, transform_augmented) # 검증은 증강 X
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 모델 준비
    model = CRNN_Metric().to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_val_loss = float('inf')
    
    print(f"학습 시작 (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            optimizer.zero_grad()
            loss = criterion(model(anchor), model(pos), model(neg))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for anchor, pos, neg in val_loader:
                anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
                val_loss += criterion(model(anchor), model(pos), model(neg)).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'crnn_metric_model_best.pth')
            print(f"  -> New Best Model Saved!")