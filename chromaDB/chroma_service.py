# chroma_service.py
import io
import os, numpy as np, pandas as pd
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from matplotlib import pyplot as plt
from starlette.responses import StreamingResponse

ROOT = os.environ.get("ROOT_DIR", ".")
CSV  = os.environ.get("CSV_PATH", "embeddings_v1_metadata.csv")
PERSIST = os.environ.get("CHROMA_DIR", "./chroma_db")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "music_v1")

app = FastAPI()
client = chromadb.PersistentClient(path=PERSIST, settings=Settings(allow_reset=False))
col = client.get_collection(COLLECTION)
meta = pd.read_csv(os.path.join(ROOT, CSV))  # title → sha256 매핑용
title2id = dict(zip(meta["title"], meta["sha256"]))
id2row = {r["sha256"]: r for r in meta.to_dict(orient="records")}

def l2n(v):
    v = np.asarray(v, dtype="float32")
    return (v / (np.linalg.norm(v) + 1e-12)).tolist()

@app.get("/")
def health():
    return {"ok": True, "collection": COLLECTION, "count": col.count()}

@app.get("/titles/suggest")
def titles_suggest(q: str, limit: int = 10):
    # 부분 일치, 대소문자 무시
    hits = meta[meta["title"].str.contains(q, case=False, na=False)].head(limit)
    return {
        "query": q,
        "count": len(hits),
        "candidates": hits[["title", "sha256"]].to_dict(orient="records")
    }


@app.get("/similar/by-title")
def similar_by_title(title: str, k: int = 10):
    if title not in title2id:
        hits = meta[meta["title"].str.contains(title, case=False, na=False)].head(10)
        return {
            "detail": "메타데이터에 해당하는 제목이 없습니다.",
            "suggest": hits["title"].to_list()
        }
        # raise HTTPException(404, detail="title not found in metadata")
    _id = title2id[title]
    # 해당 ID의 벡터를 Chroma에서 꺼낼 수도 있지만, 간단히 npy에서 로드:
    row = id2row[_id]
    import numpy as np, os
    vec = np.load(os.path.join(ROOT, row["embedding_path"])).astype("float32").ravel()
    q = l2n(vec)
    res = col.query(query_embeddings=[q], n_results=k)
    out = []
    for i, cid in enumerate(res["ids"][0]):
        md = res["metadatas"][0][i]
        out.append({"rank": i+1, "id": cid, "title": md["title"], "path": md["path"], "score": res["distances"][0][i]})
    return {"query": {"title": title}, "results": out}

@app.post("/similar/by-embedding")
def similar_by_embedding(payload: dict):
    emb = payload.get("emb")
    k = int(payload.get("k", 10))
    if not emb: raise HTTPException(400, detail="emb required")
    q = l2n(emb)
    res = col.query(query_embeddings=[q], n_results=k)
    out = []
    for i, cid in enumerate(res["ids"][0]):
        md = res["metadatas"][0][i]
        out.append({"rank": i+1, "id": cid, "title": md["title"], "path": md["path"], "score": res["distances"][0][i]})
    return {"results": out}


### 유사곡이랑 임베딩 같이 내려주기__anaPage용
@app.get("/titles/suggest_em")
def titles_suggest_em(q: str, limit: int = 10):
    # 부분 일치 검색
    hits = meta[meta["title"].str.contains(q, case=False, na=False)].head(limit)

    candidates = []

    for _, row in hits.iterrows():
        sha = row["sha256"]

        # ✅ Chroma에서 해당 sha256 메타데이터/벡터 가져오기
        res = col.get(
            where={"sha256": sha},
            include=["embeddings", "metadatas"]
        )

        if res and len(res["ids"]) > 0:
            emb = res["embeddings"][0]   # ✅ embedding 추출
        else:
            emb = None

        candidates.append({
            "title": row["title"],
            "sha256": sha,
            "embedding": emb,            # ✅ 추가됨!
        })

    return {
        "query": q,
        "count": len(candidates),
        "candidates": candidates
    }


# @app.get("/mel/{sha256}")
# def get_mel(sha256: str):
#     if sha256 not in id2row:
#         raise HTTPException(404, "sha256 not found")
#
#     row = id2row[sha256]
#     npy_path = os.path.join(ROOT, row["embedding_path"])
#     arr = np.load(npy_path)
#
#     fig = plt.figure(figsize=(6, 3))
#     plt.imshow(arr, aspect='auto', origin='lower')
#     plt.axis('off')
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
#     buf.seek(0)
#
#     return StreamingResponse(buf, media_type="image/png")


@app.get("/mel/{filename}")
def get_mel_json(filename: str):
    """
    클라이언트에서 보내준 filename 그대로 .npy 파일을 찾아서
    멜 스펙트로그램 데이터를 JSON으로 반환한다.
    """
    npy_path = os.path.join(ROOT, "embeddings_v1", f"{filename}.npy")

    if not os.path.exists(npy_path):
        raise HTTPException(404, detail=f"npy file not found: {npy_path}")

    arr = np.load(npy_path)

    # (1, n_mels, n_frames) 형태면 첫 채널만
    if arr.ndim == 3:
        arr = arr[0]

    if arr.ndim != 2:
        raise HTTPException(400, detail=f"expected 2D mel, got shape {arr.shape}")

    height, width = arr.shape
    arr = arr.astype(float)

    return {
        "filename": filename,
        "shape": [int(height), int(width)],
        "data": arr.ravel().tolist(),
    }