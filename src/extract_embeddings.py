import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
import pickle
import numpy as np

# CRNN_Metric 클래스 정의 
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
        self.rnn = nn.LSTM(input_size=8192, hidden_size=rnn_hidden_size, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.embedding = nn.Linear(rnn_hidden_size * 2, embedding_dim)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, seq_len, -1)
        r_out, _ = self.rnn(r_in)
        r_out_pooled = r_out.mean(dim=1)
        embedding = self.embedding(r_out_pooled)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

# 이미지 로딩 헬퍼 함수
# 4개의 이미지 경로 리스트를 받아 하나의 시퀀스 텐서로 만듦
def _load_segment(image_paths, transform):
    images = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            if transform:
                image = transform(image)
            images.append(image)
        except Exception as e:
            print(f"  -> 이미지 로드 오류: {img_path} ({e})")
            return None
    
    if len(images) < 4:
        return None
        
    return torch.stack(images)

if __name__ == '__main__':
    IMAGE_SIZE = (128, 128)
    EMBEDDING_DIM = 512
    MODEL_PATH = r'C:\CNN-based-Music-Classification-and-Recommendation\src\crnn_metric_model_best.pth' # 학습된 모델 경로
    DATA_DIRS = ['dataset/train', 'dataset/val'] # train, val 폴더 모두 검색
    OUTPUT_FILE = './intro_embeddings.pkl'
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 모델 로드
    model = CRNN_Metric(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # [중요] 평가 모드로 설정

    # 모든 노래 폴더 검색
    all_song_folders = []
    for data_dir in DATA_DIRS:
        folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        all_song_folders.extend(folders)
    
    print(f"총 {len(all_song_folders)}개의 노래 폴더를 찾았습니다.")

    # 임베딩 추출
    all_embeddings = {}
    
    with torch.no_grad(): # 기울기 계산 비활성화
        for song_folder in all_song_folders:
            song_name = os.path.basename(song_folder)
            
            # 1. 인트로 이미지 4장 경로 찾기 (chunk_0.png ~ chunk_3.png)
            image_paths = sorted(glob.glob(os.path.join(song_folder, '*.png')))
            
            if len(image_paths) < 4:
                print(f"  -> [{song_name}] 경고: 이미지가 4개 미만입니다. 건너뜁니다.")
                continue
            
            intro_paths = image_paths[0:4] # 0~3번 이미지를 인트로로 간주
            
            # 2. 텐서로 변환
            segment_tensor = _load_segment(intro_paths, transform)
            
            if segment_tensor is None:
                continue

            # 3. 모델 입력을 위해 배치 차원 추가 (B, Seq, C, H, W)
            segment_tensor = segment_tensor.unsqueeze(0).to(DEVICE)
            
            # 4. 임베딩 추출
            try:
                embedding = model(segment_tensor) # [1, 512]
                
                # 5. CPU로 이동 및 Numpy 변환
                embedding_np = embedding.squeeze(0).cpu().numpy()
                all_embeddings[song_name] = embedding_np
                
                if len(all_embeddings) % 100 == 0:
                    print(f"  ... {len(all_embeddings)}개 노래 처리 완료")

            except Exception as e:
                print(f"  -> [{song_name}] 임베딩 추출 오류: {e}")

    # 파일로 저장
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_embeddings, f)
        
    print(f"\n총 {len(all_embeddings)}개의 인트로 임베딩을 '{OUTPUT_FILE}'에 저장 완료")