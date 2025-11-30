import os, re, pickle, hashlib
import numpy as np
import pandas as pd

# 경로 설정
pkl_path = "intro_embeddings.pkl"     # pkl 파일 경로
out_dir = "embeddings_v1"             # npy를 저장할 폴더
os.makedirs(out_dir, exist_ok=True)

#  pkl 파일 읽기
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# 3파일명 안전하게 변환
def safe_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)  # 파일명 특수문자 제거
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200]  # 너무 긴 이름 방지

# 곡별 npy 저장 + 메타데이터 수집
rows = []
for title, vec in data.items():
    arr = np.array(vec, dtype=np.float32).ravel()
    fname = f"{safe_name(title)}.npy"
    fpath = os.path.join(out_dir, fname)
    np.save(fpath, arr)
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    rows.append({
        "title": title,
        "embedding_path": f"{out_dir}/{fname}",
        "dim": int(arr.shape[0]),
        "dtype": str(arr.dtype),
        "sha256": sha
    })

#메타데이터 CSV 생성
meta_df = pd.DataFrame(rows)
meta_df.to_csv("embeddings_v1_metadata.csv", index=False, encoding="utf-8-sig")

print(f"✅ 변환 완료! 총 {len(meta_df)}개 곡 저장됨.")
print(meta_df.head())

#

## 망함
# import os, re, pickle, hashlib
# import numpy as np
# import pandas as pd
#
# # ===== 경로 설정 =====
# EMB_PKL  = "intro_embeddings.pkl"   # 임베딩 pkl
# EMB_DIR  = "embeddings_v1"          # 임베딩 npy 저장 폴더
# MEL_DIR  = "mels_v1"                # 멜 스펙트로그램 npy 저장 폴더 (미리 정해두기)
# META_CSV = "embeddings_v1_metadata.csv"
#
# os.makedirs(EMB_DIR, exist_ok=True)
# # 멜 폴더는 나중에 별도 스크립트로 채워도 되고, 지금 미리 만들어 둬도 OK
# os.makedirs(MEL_DIR, exist_ok=True)
#
# # ===== pkl 파일 읽기 (임베딩) =====
# with open(EMB_PKL, "rb") as f:
#     data = pickle.load(f)   # { title: embedding_vector }
#
# # 파일명 안전하게 변환
# def safe_name(name: str) -> str:
#     name = re.sub(r"[\\/:*?\"<>|]+", "_", name)  # 파일명 특수문자 제거
#     name = re.sub(r"\s+", " ", name).strip()
#     return name[:200]  # 너무 긴 이름 방지
#
# rows = []
#
# for title, vec in data.items():
#     # ---- 임베딩 npy 저장 ----
#     emb_arr = np.array(vec, dtype=np.float32).ravel()
#     emb_fname = f"{safe_name(title)}.npy"
#     emb_fpath = os.path.join(EMB_DIR, emb_fname)
#     np.save(emb_fpath, emb_arr)
#
#     sha = hashlib.sha256(emb_arr.tobytes()).hexdigest()
#
#     # ---- 멜 스펙트로그램 경로 결정 ----
#     # 실제 멜 npy를 나중에 별도 코드로 만들더라도,
#     # 경로 규칙을 먼저 정해두는 느낌으로.
#     mel_fname = f"{safe_name(title)}_mel.npy"
#     mel_rel_path = f"{MEL_DIR}/{mel_fname}"
#     mel_abs_path = os.path.join(MEL_DIR, mel_fname)
#
#     # 멜 npy가 이미 존재하면 그대로 경로 사용, 없으면 빈 문자열/None
#     if os.path.exists(mel_abs_path):
#         mel_path = mel_rel_path
#     else:
#         mel_path = ""   # 또는 None
#
#     rows.append({
#         "title": title,
#         "embedding_path": f"{EMB_DIR}/{emb_fname}",
#         "dim": int(emb_arr.shape[0]),
#         "dtype": str(emb_arr.dtype),
#         "sha256": sha,
#         "mel_path": mel_path,   # ✅ 새 필드
#     })
#
# # ===== 메타데이터 CSV 생성 =====
# meta_df = pd.DataFrame(rows)
# meta_df.to_csv(META_CSV, index=False, encoding="utf-8-sig")
#
# print(f"✅ 변환 완료! 총 {len(meta_df)}개 곡 저장됨.")
# print(meta_df.head())
