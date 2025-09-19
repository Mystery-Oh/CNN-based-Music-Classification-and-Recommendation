# CNN-based-Music-Classification-and-Recommendation
CNN-based-Music-Classification-and-Recommendation
```
# 25.09.19 소은 
# ======= 파형 5가지 예제 코드 입니다. ===============
# 파이썬 3.12 // 주피터노트북 
# pip install librosa soundfile numpy matplotlib jupyterlab ipywidgets
# data 폴더는 GTZAN 

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

## 파일위치
filePath = "data/Data/genres_original/blues/blues.00000.wav"
y, sr = librosa.load(filePath)

## 파형
## 소리의 세기 변화 확인. 주파수 정보 모름. -> 원시적데이터
## x축 시간, y축 진폭(음압)
figSize = (12,4)
refValue = np.max

plt.figure(figsize=figSize)
librosa.display.waveshow(y, sr=sr)
plt.title("Wave")
plt.show()

## 스펙트로그램_STFT_주파수-시간-세기
## 시간에 따라 어떤 주파수가 얼마나 강한지?
## x 시간 , y 주파수, z 소리세기
## 타악기(드럼)dms 짧고 강한 수직패턴
## 멜로디는 부드러운 가로 패턴
## 패턴 특성 상 음악 분석, 장르 분류에 사용할 수 있음
## 짧은 시간 단위로 FFT(푸리에 변환)를 해서, 그 순간의 주파수 정보 -> 물리적인 주파수 정보
D = np.abs(librosa.stft(y))
DB = librosa.amplitude_to_db(D, ref=refValue)

plt.figure(figsize=figSize)
librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram")
plt.show()

## 멜 스펙트로그램
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=refValue)
plt.figure(figsize=figSize)
## x 시간 , y : 주파수 축을 멜 스케일로 변환. 사람의 청각 인식에 맞춘 비선형 축, z 소리세기
## 사람은 저주파는 더 민감, 고주파는 덜 민감
## 사람의 청각적 특성이 반영되었기 때문에 STFT 보다 더 적합.
## fmax : 계산할때의 최대 주파수. 이 값 이상이면 고려하지 않고 무시함. 8000이면 8000Hz 까지 사용. 8000Hz 이상의 데이터는 사용성이 글쎼..?
## 사람의 가청 주파수 : 20Hz ~ 20,000Hz(20kHz)
## sr 22050 이면 0~11025Hz
librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-Spectrogram")
plt.tight_layout()
plt.show()

# ## MFCC 13개
## 멜 스펙트로그램에서 한 단계 더 정제 추출.
## dB단위로 로그 스케일 변환
## DCT(Discrete Cosine Transform) 적용. 이는 스펙트럼의 에너지 패턴을 압축
## 압축 결과로 나오는 수가 12~20개. 오디오의 음색 특성을 얻는데 특화
## 사람이 인식하는 음성, 음색 특징이 반영되어 있음.
## 잡음에 강함.
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=figSize)
librosa.display.specshow(mfcc, x_axis="time")
plt.colorbar()
plt.title("MFCC (13)")
plt.tight_layout()
plt.show()

# ## Chromagram_화성정보
## 화음, 음계 특징을 시각화함
## C~B까지 12개 열
## 옥타브(C3,C4)등은 모두 C로 합산
## 이 곡의 두드러지는 음을 볼 수 있음.
chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
plt.figure(figsize=figSize)
librosa.display.specshow(chroma, x_axis="time", y_axis="chroma")
plt.colorbar()
plt.title("Chromagram")
plt.tight_layout()
plt.show()
```
