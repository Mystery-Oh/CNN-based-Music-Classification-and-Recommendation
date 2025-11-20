import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import Parallel, delayed
import multiprocessing

# 슬라이싱 하는 함수
def slice_audio(y, sr, clip_duration_sec=5):
    clip_samples = int(clip_duration_sec * sr) # int로 명시적 변환
    clips = []
    
    # y가 clip_samples보다 짧은 경우를 대비
    if len(y) < clip_samples:
        return []
        
    for start in range(0, len(y) - clip_samples + 1, clip_samples):
        end = start + clip_samples
        clips.append(y[start:end])
    return clips

def create_chromagram(clip, sr, hop_length=512):
    # 크로마그램 생성하는 함수
    # STFT 기반 크로마 특징 추출
    chromagram = librosa.feature.chroma_stft(y=clip, sr=sr, hop_length=hop_length)
    return chromagram

def process_mp3_to_dataset(file_path, base_output_dir):
    try:
        print(f"\n[처리 시작]: '{file_path}'")
        
        # 인트로(0-20초), 버스(20-40초), 코러스(60-80초) 정의
        segments_info = {
            "intro": (0, 20),
            "verse": (20, 40),
            "chorus": (60, 80)
        }
        max_duration_needed = 80 # 필요한 최대 시간 (코러스 끝나는 시간)
        
        file_basename = os.path.basename(file_path)
        song_folder_name = os.path.splitext(file_basename)[0]

        # 각 노래별로 별도의 폴더 생성
        song_output_dir = os.path.join(base_output_dir, song_folder_name)
        os.makedirs(song_output_dir, exist_ok=True)
        
        chunk_index = 0 # 청크 파일명에 사용할 인덱스
        
        try:
            y, sr = librosa.load(file_path, duration=max_duration_needed, sr=None) 
        except Exception as load_error:
            print(f"  -> 파일 로드 오류: {file_path}. 건너뜁니다. ({load_error})")
            return

        for segment_name, (start_sec, end_sec) in segments_info.items():
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            
            if end_sample > len(y):
                print(f"   -> 경고: '{segment_name}' 구간({start_sec}~{end_sec}초)을 처리하기에 오디오가 너무 짧습니다 ({len(y)/sr:.2f}초). 이 파일을 건너뜁니다.")
                return

            y_segment = y[start_sample:end_sample]

            # 5초 단위로 오디오 슬라이싱
            audio_clips = slice_audio(y_segment, sr, clip_duration_sec=5)
            
            # 20초 구간에서 정확히 4개의 클립이 나오는지 확인
            if len(audio_clips) != 4:
                print(f"  -> 경고: {segment_name} 구간에서 4개가 아닌 {len(audio_clips)}개의 클립이 생성되었습니다 ({len(y_segment)/sr:.2f}초 길이). 이 파일을 건너뜁니다.")
                return

            # 이미지를 생성하고 저장
            for clip in audio_clips:
                chromagram = create_chromagram(clip, sr) # sr을 전달

                plt.figure(figsize=(4.32, 2.88)) # 128x128 리사이즈를 고려한 비율
                librosa.display.specshow(chromagram, sr=sr, hop_length=512, y_axis='chroma')
                plt.axis('off')
                
                save_path = os.path.join(song_output_dir, f"chunk_{chunk_index}.png")
                
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                chunk_index += 1
        
        labels = [0]*4 + [1]*4 + [2]*4
        label_string = ",".join(map(str, labels))
        with open(os.path.join(song_output_dir, 'labels.txt'), 'w') as f:
            f.write(label_string)
            
        print(f"  -> 총 {chunk_index}개의 크로마그램 이미지와 labels.txt를 성공적으로 저장했습니다. ('{file_path}')")

    except Exception as e:
        # 병렬 처리 시 오류 출력이 섞일 수 있으나, 어떤 파일에서 문제 생겼는지 확인 가능
        print(f"  -> [심각한 오류] '{file_path}' 처리 중 예외 발생: {e}")

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
        
        total_cores = multiprocessing.cpu_count()
        # total_cores 조절로 사용하는 CPU 코어 수 지정
        num_cores_to_use = max(1, total_cores - 2) 
        
        print(f"총 CPU 코어: {total_cores}개")
        print(f"시스템 응답성을 위해 2개를 제외한 {num_cores_to_use}개의 코어를 사용하여 병렬 처리를 시작합니다...")

        Parallel(n_jobs=num_cores_to_use)(
            delayed(process_mp3_to_dataset)(mp3_file_path, train_data_dir) 
            for mp3_file_path in found_mp3_files
        )
 
        print("\n모든 작업이 완료되었습니다.")