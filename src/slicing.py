import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def slice_audio(y, sr, clip_duration_sec=5):
    """5초 단위로 오디오를 슬라이스하는 함수"""
    clip_samples = clip_duration_sec * sr
    clips = []
    for start in range(0, len(y), clip_samples):
        end = start + clip_samples
        if end <= len(y):
            clips.append(y[start:end])
    return clips

def create_spectrogram(clip, sr, n_mels=128, hop_length=512): # n_mels와 hop_length 조정
    """클립으로 멜 스펙트로그램을 생성하는 함수"""
    S = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def process_mp3_to_dataset(file_path, base_output_dir):
    try:
        print(f"\n[처리 시작]: '{file_path}'")
        
        # 인트로(0-20초), 버스(20-40초), 코러스(60-80초) 정의
        # 각 구간은 4개의 5초 클립으로 구성됨
        time_segments = [(0, 20), (20, 40), (60, 80)]
        
        file_basename = os.path.basename(file_path)
        song_folder_name = os.path.splitext(file_basename)[0]

        # [수정] 각 노래별로 별도의 폴더 생성
        song_output_dir = os.path.join(base_output_dir, song_folder_name)
        os.makedirs(song_output_dir, exist_ok=True)
        
        chunk_index = 0 # 청크 파일명에 사용할 인덱스

        # 지정된 각 시간 구간에 대해 반복 처리
        for segment_start_sec, segment_end_sec in time_segments:
            duration_to_load = segment_end_sec - segment_start_sec
            
            # 오디오의 특정 구간 불러오기
            y, sr = librosa.load(file_path, offset=segment_start_sec, duration=duration_to_load)

            # 5초 단위로 오디오 슬라이싱
            audio_clips = slice_audio(y, sr, clip_duration_sec=5)
            
            # 20초 구간에서 정확히 4개의 클립이 나오는지 확인
            if len(audio_clips) != 4:
                print(f"  -> 경고: {segment_start_sec}~{segment_end_sec}초 구간에서 4개가 아닌 {len(audio_clips)}개의 클립이 생성되었습니다. 이 파일을 건너뜁니다.")
                return # 이 파일 처리를 중단

            # 이미지를 생성하고 저장
            for clip in audio_clips:
                S_dB = create_spectrogram(clip, sr)

                plt.figure(figsize=(4.32, 2.88)) # 128x128 리사이즈를 고려한 비율
                librosa.display.specshow(S_dB, sr=sr, hop_length=512)
                plt.axis('off')
                
                # [수정] 순차적인 파일명 사용 (chunk_0.png, chunk_1.png, ...)
                save_path = os.path.join(song_output_dir, f"chunk_{chunk_index}.png")
                
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                chunk_index += 1
        
        # 고정된 라벨 파일 생성
        labels = [0]*4 + [1]*4 + [2]*4
        label_string = ",".join(map(str, labels))
        with open(os.path.join(song_output_dir, 'labels.txt'), 'w') as f:
            f.write(label_string)
            
        print(f"  -> 총 {chunk_index}개의 스펙트로그램 이미지와 labels.txt를 성공적으로 저장했습니다.")

    except Exception as e:
        print(f"  -> 오류 발생: '{file_path}' 처리 중 문제가 발생했습니다. 원인: {e}")

if __name__ == "__main__":
    root_dir_to_search = "." # MP3 파일을 찾을 시작 폴더
    train_data_dir = os.path.join("dataset", "train") # 최종 데이터셋이 저장될 위치
    
    os.makedirs(train_data_dir, exist_ok=True)

    print(f"'{root_dir_to_search}' 폴더 및 모든 하위 폴더에서 MP3 파일을 검색합니다...")
    
    found_mp3_files = []
    for dirpath, _, filenames in os.walk(root_dir_to_search):
        for filename in filenames:
            if filename.lower().endswith(".mp3"):
                full_path = os.path.join(dirpath, filename)
                found_mp3_files.append(full_path)

    if not found_mp3_files:
        print("처리할 MP3 파일을 찾지 못했습니다.")
    else:
        print(f"총 {len(found_mp3_files)}개의 MP3 파일을 찾았습니다. 데이터셋 생성을 시작합니다.")
        for mp3_file_path in found_mp3_files:
            process_mp3_to_dataset(mp3_file_path, train_data_dir)
        print("\n모든 작업이 완료되었습니다.")