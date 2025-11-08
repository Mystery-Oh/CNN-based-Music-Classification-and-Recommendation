import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import random

# --- 1. [수정] 데이터 증강(Augmentation)이 포함된 Transform 정의 ---
# 학습 시 Anchor/Negative에 적용할 기본 변환
transform_basic = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 학습 시 Positive 샘플에만 적용할 '데이터 증강' 변환
# (원본과 약간 다르게 보이도록 노이즈 추가)
transform_augmented = transforms.Compose([
    transforms.Resize((128, 128)),
    # [추가] 약간의 색상, 밝기 변형
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # [추가] 5% 확률로 수평 뒤집기 (의미가 있을지는 모르나 일반적인 증강)
    transforms.RandomHorizontalFlip(p=0.05),
    # [추가] 10% 확률로 랜덤하게 10% 픽셀 가리기 (cutout)
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# --- 2. [수정] Dataset 클래스 (데이터 증강 적용) ---
class TripletSongSegmentDataset(Dataset):
    def __init__(self, data_dir, transform_basic, transform_augmented):
        self.data_dir = data_dir
        # [수정] 2개의 transform을 받음
        self.transform_basic = transform_basic
        self.transform_augmented = transform_augmented 
        
        self.song_folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        self.intros = []
        print("데이터셋 스캔 중 (인트로만 수집)...")
        for song_folder in self.song_folders:
            image_paths = sorted(glob.glob(os.path.join(song_folder, '*.png')))
            if len(image_paths) >= 4:
                self.intros.append(image_paths[0:4])
        print(f"총 {len(self.intros)}개의 인트로 세그먼트 발견.")

    def __len__(self):
        return len(self.intros)

    def _load_segment(self, image_paths, transform_to_apply):
        # [수정] 어떤 transform을 적용할지 인자로 받음
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            # [수정] 지정된 transform 적용
            image = transform_to_apply(image)
            images.append(image)
        return torch.stack(images)

    def __getitem__(self, idx):
        # 1. Anchor (기준): idx번째 인트로 (기본 변환)
        anchor_paths = self.intros[idx]
        anchor = self._load_segment(anchor_paths, self.transform_basic)
        
        # --- [핵심 수정] ---
        # 2. Positive (긍정): *똑같은 idx번째* 인트로 (증강 변환)
        # (Anchor와 Positive는 같은 원본 이미지이지만, 증강 때문에 텐서 값은 다름)
        positive_paths = self.intros[idx] 
        positive = self._load_segment(positive_paths, self.transform_augmented)
        
        # 3. Negative (부정): *다른* 인트로 (기본 변환)
        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.intros) - 1)
        negative_paths = self.intros[negative_idx]
        negative = self._load_segment(negative_paths, self.transform_basic)
        
        return anchor, positive, negative

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

if __name__ == '__main__':
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 32 # (이전 대화에서 32로 늘렸으므로 유지)
    EPOCHS = 50
    EMBEDDING_DIM = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ... (GPU 확인, num_workers 설정 등은 동일) ...
    print(f"현재 학습에 사용 중인 장치: {DEVICE}")
    total_cores = os.cpu_count() or 1
    num_workers = max(1, total_cores - 2)
    print(f"데이터 로딩에 {num_workers}개의 CPU 워커를 사용합니다.")


    # [수정] Dataset 생성 시 2개의 transform 전달
    train_dataset = TripletSongSegmentDataset(
        data_dir='dataset/train', 
        transform_basic=transform_basic, 
        transform_augmented=transform_augmented
    )
    
    # [수정] DataLoader 생성 (num_workers, pin_memory 적용)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # [수정] 검증(Validation) 데이터 로더 추가
    # (검증 시에는 데이터 증강을 사용하지 않음)
    val_dataset = TripletSongSegmentDataset(
        data_dir='dataset/val', 
        transform_basic=transform_basic, 
        transform_augmented=transform_basic # [중요] 검증 시에는 증강 안 함
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"훈련 데이터: {len(train_dataset)}개 세트, 검증 데이터: {len(val_dataset)}개 세트")

    # ... (모델, 손실함수, 옵티마이저 설정은 동일) ...
    model = CRNN_Metric(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ... (최적 모델 저장을 위한 변수) ...
    best_val_loss = float('inf')

    print("거리 학습(Metric Learning) 시작 (데이터 증강 ver.)...")
    
    for epoch in range(EPOCHS):
        # --- 훈련(Training) 단계 ---
        model.train()
        total_train_loss = 0
        
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- 검증(Validation) 단계 ---
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
                
                emb_anchor = model(anchor)
                emb_positive = model(positive) # 검증 시 A, P는 동일한 원본 (증강 X)
                emb_negative = model(negative)
                
                loss = criterion(emb_anchor, emb_positive, emb_negative)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        # 에포크 결과 출력
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'crnn_metric_model_best.pth')
            print(f"  -> New best model saved with val_loss: {best_val_loss:.4f}")

    print("학습 완료")