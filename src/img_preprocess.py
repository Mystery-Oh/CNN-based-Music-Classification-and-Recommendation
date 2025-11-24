import torch
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm


# 원본 데이터 경로
DATA_DIRS = ['dataset_chroma/train', 'dataset_chroma/val']

# 전처리된 데이터가 저장될 경로
OUTPUT_DIR_BASE = 'dataset_preprocessed_chroma'

IMAGE_SIZE = (128, 128)

pre_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(), 
])

def process_and_save(data_dirs):
    print("오프라인 전처리를 시작합니다...")
    
    for data_dir in data_dirs:
        # 저장할 폴더 경로 생성 (예: dataset_preprocessed/train)
        output_dir = data_dir.replace('dataset', OUTPUT_DIR_BASE)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[{data_dir}] -> [{output_dir}] 변환 중...")
        
        # 모든 노래 폴더 검색
        song_folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        success_count = 0
        
        for song_folder in tqdm(song_folders):
            song_name = os.path.basename(song_folder)
            
            # 이미지 파일 찾기
            image_paths = sorted(glob.glob(os.path.join(song_folder, '*.png')))
            
            # 인트로(0~3번) 이미지가 있는지 확인
            if len(image_paths) < 4:
                continue
                
            intro_paths = image_paths[0:4]
            
            try:
                # 이미지 4장을 열어서 텐서로 변환 후 스택
                images = []
                for img_path in intro_paths:
                    image = Image.open(img_path).convert("RGB")
                    image = pre_transform(image)
                    images.append(image)
                
                # [4, 3, 128, 128] 크기의 텐서 생성
                tensor_data = torch.stack(images)
                
                # .pt 파일로 저장 (파일명: 노래제목_intro.pt)
                save_path = os.path.join(output_dir, f"{song_name}_intro.pt")
                torch.save(tensor_data, save_path)
                
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {song_name}: {e}")
                
        print(f"  -> {success_count}개의 파일 생성 완료.")

if __name__ == '__main__':
    process_and_save(DATA_DIRS)