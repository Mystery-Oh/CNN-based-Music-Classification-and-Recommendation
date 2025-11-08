import pickle
import numpy as np
import os
import glob
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_spectrogram(image_path, image_size=(128, 128)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), 

    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def plot_spectrogram_comparison(query_song_name, recommended_songs_info, dataset_base_path, image_size=(128, 128)):

    # 총 12개의 이미지 (인트로 4, 버스 4, 코러스 4)
    num_chunks_per_song = 12 
    
    # 전체 플롯의 크기 조정
    fig, axes = plt.subplots(
        len(recommended_songs_info) + 1,  # 쿼리 곡 + 추천 곡 개수
        num_chunks_per_song,              # 각 곡의 청크 개수 (12개)
        figsize=(num_chunks_per_song * 1.5, (len(recommended_songs_info) + 1) * 1.5) # 개당 1.5x1.5인치
    )
    
    if len(recommended_songs_info) == 0: # 추천곡이 없는 경우 예외 처리
        print("추천할 곡이 없어 스펙트로그램을 비교할 수 없습니다.")
        return

    # axes가 2D 배열이 아닐 수 있으므로 항상 2D로 만듦
    if len(recommended_songs_info) == 0:
        axes = np.array([axes])
    elif len(recommended_songs_info) == 1:
        axes = np.array([axes[0], axes[1]]) # 쿼리 + 1개 추천곡

    # 쿼리 곡과 추천 곡의 라벨을 이름과 함께 두 줄로 표시하도록 수정
    query_label = f"{query_song_name}\n(쿼리 곡)"
    recommended_tuples = [(name, f"{name}\n(유사도: {sim:.4f})") for name, sim in recommended_songs_info]

    all_songs_to_plot = [(query_song_name, query_label)] + recommended_tuples

    for row_idx, (song_name, title_prefix) in enumerate(all_songs_to_plot):
        song_folder_path = os.path.join(dataset_base_path, song_name)
        
        # 청크 이미지 경로 가져오기
        image_paths = sorted(glob.glob(os.path.join(song_folder_path, 'chunk_*.png')))
        
        # 12개 이미지가 모두 있는지 확인
        if len(image_paths) != num_chunks_per_song:
            print(f"경고: '{song_name}' 폴더에 {num_chunks_per_song}개의 청크 이미지가 없습니다. 스킵합니다.")
            continue

        for col_idx in range(num_chunks_per_song):
            ax = axes[row_idx, col_idx]
            img_tensor = load_and_preprocess_spectrogram(image_paths[col_idx], image_size)
            
            # Matplotlib은 (H, W, C) 형태의 이미지를 선호하므로, (C, H, W)를 변환
            ax.imshow(img_tensor.permute(1, 2, 0)) 
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col_idx == 0: # 첫 번째 열에만 곡 이름/유사도 표시
                ax.set_ylabel(title_prefix, rotation=0, ha='right', va='center', fontsize=10)
            
            # 섹션 표시 (인트로, 버스, 코러스)
            if row_idx == 0: # 쿼리 곡 상단에만 섹션명 표시
                if col_idx == 0: ax.set_title("인트로", fontsize=9)
                elif col_idx == 4: ax.set_title("버스", fontsize=9)
                elif col_idx == 8: ax.set_title("코러스", fontsize=9)

    plt.tight_layout()
    plt.suptitle(f"'{query_song_name}'와 유사곡 스펙트로그램 비교", y=1.02, fontsize=14)
    plt.show()

class Recommender:
    def __init__(self, signature_db_path):
        with open(signature_db_path, 'rb') as f:
            self.song_signatures = pickle.load(f)
        
        self.song_names = list(self.song_signatures.keys())
        self.signature_vectors = np.array(list(self.song_signatures.values()))
        
        self.search_index = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.search_index.fit(self.signature_vectors)

    def recommend(self, song_name, top_n=10):
        if song_name not in self.song_signatures:
            return f"'{song_name}'을(를) DB에서 찾을 수 없습니다."
        
        query_vector = self.song_signatures[song_name].reshape(1, -1)
        
        distances, indices = self.search_index.kneighbors(query_vector)
        
        recommendations = []
        for i in range(1, top_n + 1):
            if i >= len(indices[0]): break
            song_index = indices[0][i]
            similar_song = self.song_names[song_index]
            similarity = 1 - distances[0][i]
            recommendations.append((similar_song, similarity))
            
        return recommendations

if __name__ == '__main__':
    recommender = Recommender('song_signatures.pkl')
    
    if recommender.song_names:
        # 추천받고 싶은 곡을 직접 지정하거나, DB에서 첫 번째 곡 선택
        song_to_recommend = recommender.song_names[0] # 예시: DB의 첫 번째 곡을 쿼리 곡으로 사용
        # song_to_recommend = "원하는_곡_이름" # 특정 곡을 쿼리하고 싶다면 이렇게 변경
        
        recommendations = recommender.recommend(song_to_recommend, top_n=5) # 상위 5개 추천
        
        print(f"'{song_to_recommend}'와 비슷한 곡 Top {len(recommendations)}개:")
        for song, sim in recommendations:
            print(f"- {song} (유사도: {sim:.4f})")
            
        # dataset/train 폴더의 실제 경로를 지정해야 합니다.
        spectrogram_data_path = "dataset/train" 
        plot_spectrogram_comparison(song_to_recommend, recommendations, spectrogram_data_path)
        
    else:
        print("DB에 곡이 없습니다.")