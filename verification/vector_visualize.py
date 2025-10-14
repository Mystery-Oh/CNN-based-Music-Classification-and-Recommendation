import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

"""
    CNN 결과값 벡터 시각화 코드
"""

try:
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
except:
    plt.rcParams['font.family'] = 'AppleGothic' # macOS
plt.rcParams['axes.unicode_minus'] = False

# 저장된 시그니처 벡터 DB 불러오기
with open('song_signatures.pkl', 'rb') as f:
    song_signatures = pickle.load(f)

song_names = list(song_signatures.keys())
signature_vectors = np.array(list(song_signatures.values()))

# t-SNE 모델 생성 및 차원 축소 실행
print("t-SNE 차원 축소를 시작합니다")
tsne = TSNE(n_components=2,          # 2차원으로 축소
            perplexity=30.0,         # 데이터 포인트 주변의 이웃 수. 보통 5~50 사이 값 사용
            learning_rate='auto',    # 학습률
            n_iter=1000,             # 반복 횟수
            random_state=42)         # 결과를 일정하게 유지하기 위한 시드
            
vectors_2d = tsne.fit_transform(signature_vectors)

# Matplotlib으로 2D 산점도(Scatter Plot) 시각화
plt.figure(figsize=(14, 10))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)

# 각 점에 노래 제목 라벨 추가
for i, name in enumerate(song_names):
    plt.annotate(name, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=8)

plt.title("t-SNE 벡터 시각화", fontsize=16)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()