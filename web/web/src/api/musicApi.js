// src/api/musicApi.js
const BASE_URL = "http://localhost:4000/";

// 공통 GET 헬퍼
async function get(path, params = {}) {
    const url = new URL(path, BASE_URL);

    console.log(params);
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") {
            url.searchParams.append(key, value);
        }
    });

    const res = await fetch(url.toString());
    if (!res.ok) {
        throw new Error(`GET ${path} 실패: ${res.status}`);
    }
    return res.json();
}

// 공통 POST 헬퍼 (JSON)
async function post(path, body) {
    const res = await fetch(`${BASE_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        throw new Error(`POST ${path} 실패: ${res.status}`);
    }
    return res.json();
}


// 서버 상태 체크 (GET /server)
export function pingServer() {
    return get("/");
}

// 트랙 리스트 (GET /trakList)
export function fetchTrackList(page = 1, limit = 20) {
    return get("/trakList", { page, limit });
}

// 검색 (GET /search?q=...)
export function searchTracks(q, page = 1, limit = 20) {
    return get("/search", { q, page, limit });
}

// CSV 업데이트 (POST /csvUpdate)
export function csvUpdate() {
    return post("/csvUpdate", {}); // 필요하면 body 넣기
}

// CSV 업로드 폼 전송 (POST /update-csv-upload)
export async function uploadCsv(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${BASE_URL}/update-csv-upload`, {
        method: "POST",
        body: formData,
    });
    if (!res.ok) {
        throw new Error(`CSV 업로드 실패: ${res.status}`);
    }
    return res.json();
}

//크로마
// GET http://localhost:4000/api/similar?title=akmu&k=5
export function fetchChromaSimilarity(title, k = 5) {
    return get("/api/similar", { title, k });
}

// GET http://localhost:4000/api/similar/suggest?q=Rose
export function fetchChromaSuggest(q) {
    return get("/api/similar/suggest", { q });
}

//youtube 검새ㅐㄱ
// GET /api/music/youtube/search?q=...
export function fetchYoutubeByQuery(q) {
    return get("/api/youtube/search", { q });
}
