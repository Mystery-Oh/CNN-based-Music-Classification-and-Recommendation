import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import random

class TripletSongSegmentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.song_folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        self.intros = []
        # [수정] self.non_intros 리스트가 더 이상 필요 없으므로 제거 (혹은 비워둠)
        # self.non_intros = [] 

        print("데이터셋 스캔 중 (인트로만 수집)...")
        for song_folder in self.song_folders:
            image_paths = sorted(glob.glob(os.path.join(song_folder, '*.png')))
            
            if len(image_paths) >= 4: # 최소 4개의 이미지가 있는지 확인 (인트로)
                # 인트로 세그먼트 (4개 이미지 경로)
                self.intros.append(image_paths[0:4])
            
            # [수정] 비-인트로 세그먼트는 더 이상 수집하지 않음
            
        print(f"총 {len(self.intros)}개의 인트로 세그먼트 발견.")

    def __len__(self):
        # 학습은 '인트로'의 개수를 기준으로 함
        return len(self.intros)

    def _load_segment(self, image_paths):
        # 4개의 이미지 경로 리스트를 받아 하나의 시퀀스 텐서로 만듦
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        # [4, 3, 128, 128] 크기의 텐서 반환
        return torch.stack(images)

    def __getitem__(self, idx):
        # 1. Anchor (기준): idx번째 인트로
        anchor_paths = self.intros[idx]
        
        # 2. Positive (긍정): Anchor가 아닌 다른 인트로 (랜덤 선택)
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.randint(0, len(self.intros) - 1)
        positive_paths = self.intros[positive_idx]
        
        # --- [핵심 수정] ---
        # 3. Negative (부정): Anchor 및 Positive와 모두 다른 인트로 (랜덤 선택)
        negative_idx = idx
        while negative_idx == idx or negative_idx == positive_idx:
            negative_idx = random.randint(0, len(self.intros) - 1)
        negative_paths = self.intros[negative_idx] # self.non_intros 대신 self.intros 사용
        
        # 각 세그먼트를 텐서로 로드
        anchor = self._load_segment(anchor_paths)
        positive = self._load_segment(positive_paths)
        negative = self._load_segment(negative_paths)
        
        return anchor, positive, negative

# --- 2. CRNN 모델 (변경 없음) ---
class CRNN_Metric(nn.Module):
    def __init__(self, rnn_hidden_size=256, embedding_dim=512):
        super(CRNN_Metric, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )
        # 128x128 이미지 기준 CNN 출력 크기: 128 * 8 * 8 = 8192
        self.rnn = nn.LSTM(input_size=8192, hidden_size=rnn_hidden_size, num_layers=2, 
                           batch_first=True, bidirectional=True)
        
        self.embedding_layer = nn.Linear(rnn_hidden_size * 2, embedding_dim)

    def forward(self, x):
        # x shape: [B, SeqLen, C, H, W] (e.g., [B, 4, 3, 128, 128])
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        c_out = self.cnn(c_in) # [B*SeqLen, 128, 8, 8]
        
        r_in = c_out.view(batch_size, seq_len, -1) # [B, SeqLen, 8192]
        
        r_out, _ = self.rnn(r_in) # [B, SeqLen, Hidden*2]
        
        # 시퀀스 전체의 특징을 대표하기 위해 평균 풀링
        r_out_pooled = r_out.mean(dim=1) # [B, Hidden*2]
        
        embedding = self.embedding_layer(r_out_pooled) # [B, embedding_dim]
        
        # L2 정규화
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

# --- 3. 학습 스크립트 (DataLoader 최적화 적용) ---
if __name__ == '__main__':
    # 하이퍼파라미터
    IMAGE_SIZE = (128, 128)
    # [권장] 4070Ti VRAM이면 4는 매우 작습니다. 
    # 메모리 부족(OOM) 오류가 나지 않는 선에서 16, 32, 64 등으로 늘려보세요.
    BATCH_SIZE = 32
    EPOCHS = 50
    EMBEDDING_DIM = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [수정] GPU 사용 확인
    print(f"현재 학습에 사용 중인 장치: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")

    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # [수정] DataLoader 최적화
    # 사용할 CPU 코어 수 결정 (시스템 응답성을 위해 2개 제외)
    total_cores = os.cpu_count() or 1 # os.cpu_count()가 None일 경우 1을 사용
    num_workers = max(1, total_cores-2) # 최소 1개의 워커는 보장
    print(f"데이터 로딩에 {num_workers}개의 CPU 워커를 사용합니다.")

    train_dataset = TripletSongSegmentDataset(data_dir='dataset/train', transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,  # <- 병렬 데이터 로딩 (핵심)
        pin_memory=True           # <- CPU -> GPU 전송 속도 향상 (num_workers > 0일 때 권장)
    )
    
    # 모델, 손실함수, 옵티마이저
    model = CRNN_Metric(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("거리 학습(Metric Learning) 시작...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # 이제 GPU가 연산하는 동안 CPU 워커들이 백그라운드에서 다음 배치를 준비합니다.
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'crnn_metric_model.pth')
    print("임베딩 모델 저장 완료")