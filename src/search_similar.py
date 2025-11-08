import torch
import pickle
import numpy as np

EMBEDDING_FILE = 'intro_embeddings.pkl'
TOP_K = 5 # 상위 5개 결과
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print(f"'{EMBEDDING_FILE}'에서 임베딩 벡터를 불러오는 중...")
try:
    with open(EMBEDDING_FILE, 'rb') as f:
        all_embeddings_dict = pickle.load(f)
except FileNotFoundError:
    print(f"오류: '{EMBEDDING_FILE}'을 찾을 수 없습니다.")
    print("먼저 `extract_embeddings.py`를 실행하여 임베딩 파일을 생성하세요.")
    exit()

# 검색을 위해 (노래 이름 리스트)와 (벡터 텐서)로 분리
song_names = list(all_embeddings_dict.keys())
vectors = list(all_embeddings_dict.values())

# (최적화) 모든 벡터를 GPU 텐서로 미리 변환
vectors_tensor = torch.tensor(np.array(vectors)).to(DEVICE)
print(f"총 {len(song_names)}개의 임베딩을 GPU로 로드했습니다.")

def find_similar_intros(query_song_name, top_k=TOP_K):
    """
    특정 노래 이름이 주어지면, 가장 유사한 인트로를 가진 Top-K 곡을 찾습니다.
    """
    
    # 1. 쿼리 벡터 찾기
    if query_song_name not in all_embeddings_dict:
        print(f"오류: '{query_song_name}'이(가) 임베딩 데이터베이스에 없습니다.")
        # 데이터베이스에 있는 노래 이름 중 하나를 무작위로 제안
        if song_names:
            print(f"  -> 예시 노래 이름: '{song_names[0]}'")
        return

    query_vector = all_embeddings_dict[query_song_name]
    query_tensor = torch.tensor(query_vector).to(DEVICE)

    # 2. 코사인 유사도 계산 (모델이 L2 정규화를 했으므로, 내적(matmul)이 코사인 유사도임)
    # (GPU에서 모든 벡터와 쿼리 벡터 간의 내적을 한 번에 계산)
    similarities = torch.matmul(vectors_tensor, query_tensor)
    
    # 3. 점수가 높은 순서대로 정렬
    # (top_k + 1)을 하는 이유: 자기 자신(유사도 1.0)이 1위로 나오기 때문
    scores, indices = torch.topk(similarities, top_k + 1)
    
    # CPU로 결과 이동
    scores = scores.cpu().numpy()
    indices = indices.cpu().numpy()
    
    print("-" * 30)
    print(f"'{query_song_name}'와(과) 유사한 곡 Top {top_k}:")
    print("-" * 30)
    
    # 0번 인덱스(자기 자신)를 건너뛰고 1번부터 출력
    for i in range(1, top_k + 1):
        if i >= len(indices):
            break
            
        song_name = song_names[indices[i]]
        score = scores[i]
        
        # score가 1.0에 가까울수록 유사함
        print(f"  {i}. {song_name} (유사도: {score:.4f})")
    print("-" * 30)

if __name__ == '__main__':
    if not song_names:
        print("임베딩이 비어있어 검색을 실행할 수 없습니다.")
    else:
        # --- [수정된 부분 시작] ---
        
        # 1. 사용자에게 검색할 노래 이름을 입력받음
        print("\n--- 유사도 검색 ---")
        print(f"데이터베이스에 총 {len(song_names)}개의 곡이 있습니다.")
        print(f"예: '{song_names[0]}' 또는 '{song_names[1]}'")
        
        # 사용자가 중단할 때까지 계속 검색을 요청
        while True: 
            QUERY_SONG = input("\n검색할 노래의 폴더명을 입력하세요 (종료하려면 엔터): ")
            
            # 2. 사용자가 그냥 엔터를 누르면 프로그램 종료
            if not QUERY_SONG:
                print("검색을 종료합니다.")
                break
                
            # 3. 입력받은 이름으로 검색 함수 호출
            find_similar_intros(QUERY_SONG)
            
        # --- [수정된 부분 끝] ---