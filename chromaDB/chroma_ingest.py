## 소은 적재할 때 한번만 쓰세요
import os, math, numpy as np, pandas as pd
import chromadb
from chromadb.config import Settings

ROOT = os.environ.get("ROOT_DIR", ".")                 # CSV와 npy 경로 기준 루트
CSV  = os.environ.get("CSV_PATH", "embeddings_v1_metadata.csv")
PERSIST = os.environ.get("CHROMA_DIR", "./chroma_db")  # 영구 보관 디렉토리
COLLECTION = os.environ.get("CHROMA_COLLECTION", "music_v1")

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def ingest(batch=1000):
    client = chromadb.PersistentClient(path=PERSIST, settings=Settings(allow_reset=False))
    try:
        col = client.get_collection(COLLECTION)
    except:
        col = client.create_collection(COLLECTION, metadata={"hnsw:space":"cosine"})

    meta = pd.read_csv(os.path.join(ROOT, CSV))
    ## 혹시 중복곡 있으면 제거하기
    meta = meta.drop_duplicates(subset=["sha256"])
    ids, embs, metadatas = [], [], []

    for i, row in meta.iterrows():
        p = os.path.join(ROOT, row["embedding_path"])
        arr = np.load(p).astype("float32").ravel()
        ids.append(row["sha256"])  # 고유 ID
        embs.append(arr.tolist())
        metadatas.append({
            "title": row["title"],
            "path": row["embedding_path"],
            "dim": int(row["dim"]),
            "dtype": str(row["dtype"])
        })

        if len(ids) >= batch:
            X = np.array(embs, dtype="float32")
            X = l2_normalize(X)
            col.add(ids=ids, embeddings=X.tolist(), metadatas=metadatas)
            ids, embs, metadatas = [], [], []
            print(f"added {i+1} rows")

    if ids:
        X = np.array(embs, dtype="float32")
        X = l2_normalize(X)
        col.add(ids=ids, embeddings=X.tolist(), metadatas=metadatas)
    print("✅ ingest finished")

if __name__ == "__main__":
    ingest()
