import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from train import CRNN, SongSequenceDataset # train.py에서 정의한 클래스 및 변환 재사용
import os

# 특징 추출기 모델 정의
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.cnn = original_model.cnn
        self.rnn = original_model.rnn

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, seq_len, -1)
        r_out, _ = self.rnn(r_in)
        return r_out

if __name__ == '__main__':
    # 학습된 모델 로드 및 특징 추출기 생성
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    original_model = CRNN(num_classes=3)
    original_model.load_state_dict(torch.load('crnn_feature_extractor.pth'))
    
    feature_extractor = FeatureExtractor(original_model).to(DEVICE)
    feature_extractor.eval()
    
    # 모든 곡에 대해 시그니처 벡터 추출
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = SongSequenceDataset(data_dir='dataset/train', transform=transform)
    # 배치 크기를 1로 하여 한 곡씩 처리
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    song_signatures = {}
    
    print("시그니처 벡터 추출 시작...")
    with torch.no_grad():
        for i, (sequence, _) in enumerate(loader):
            song_name = os.path.basename(dataset.song_folders[i])
            sequence = sequence.to(DEVICE)
            
            # (1, 12, 512) 크기의 특징 벡터 시퀀스 추출 후 squeeze로 (12, 512)로 변경
            feature_sequence = feature_extractor(sequence).squeeze(0)
            
            # 고정된 인덱스로 각 섹션의 벡터들을 슬라이싱
            intro_vectors = feature_sequence[0:4]
            verse_vectors = feature_sequence[4:8]
            chorus_vectors = feature_sequence[8:12]
            
            # 평균 풀링으로 각 섹션의 대표 벡터 생성
            intro_vec = intro_vectors.mean(dim=0)
            verse_vec = verse_vectors.mean(dim=0)
            chorus_vec = chorus_vectors.mean(dim=0)
            
            # 세 벡터를 결합하여 최종 시그니처 벡터 생성
            signature_vector = torch.cat([intro_vec, verse_vec, chorus_vec])
            
            song_signatures[song_name] = signature_vector.cpu().numpy()

    # 추출된 시그니처 벡터 저장 
    with open('song_signatures.pkl', 'wb') as f:
        pickle.dump(song_signatures, f)
        
    print(f"{len(song_signatures)}개 곡의 시그니처 벡터 저장 완fy")