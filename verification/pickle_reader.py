import pickle
import numpy as np

# 확인할 파일 경로
FILE_PATH = 'song_signatures.pkl'

try:
    # 파일을 바이너리 읽기 모드('rb')로 열기
    with open(FILE_PATH, 'rb') as f:
        # pickle.load()로 파일에 저장된 파이썬 객체 불러오기
        data = pickle.load(f)

    print(f"✅ '{FILE_PATH}' 파일을 성공적으로 불러왔습니다.\n")

    # 1. 저장된 데이터의 전체 타입 확인 (Dictionary여야 함)
    print(f"파일의 전체 데이터 타입: {type(data)}")

    # 2. 저장된 총 곡의 수 확인
    if isinstance(data, dict):
        print(f"저장된 총 곡의 수: {len(data)}개")
        
        song_names = list(data.keys())
        if song_names:
            # 3. 어떤 곡들이 저장되었는지 앞 5개 샘플 확인
            print(f"저장된 곡 이름 샘플: {song_names[:5]}")
            
            # 4. 첫 번째 곡의 시그니처 벡터 상세 정보 확인
            first_song_name = song_names[0]
            first_vector = data[first_song_name]
            
            print(f"\n--- '{first_song_name}'의 데이터 분석 ---")
            print(f"벡터의 타입: {type(first_vector)}")
            print(f"벡터의 차원(Shape): {first_vector.shape}")
            print(f"벡터의 값 일부 (앞 10개): \n{first_vector[:10]}")
        else:
            print("파일 안에 저장된 곡이 없습니다.")
    else:
        print("예상했던 딕셔너리(Dictionary) 타입이 아닙니다.")

except FileNotFoundError:
    print(f" 오류: '{FILE_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f" 파일을 읽는 중 오류가 발생했습니다: {e}")