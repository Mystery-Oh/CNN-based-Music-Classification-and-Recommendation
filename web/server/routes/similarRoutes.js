const express = require("express");
const router = express.Router();

const CHROMA_URL = process.env.CHROMA_URL || "http://localhost:8001";

//검색제안
router.get("/similar/suggest", async (req, res, next) => {
    try {
        const q = (req.query.q || "").trim();
        if (!q) return res.status(400).json({ error: "q required" });
        const r = await fetch(`${process.env.CHROMA_URL}/titles/suggest?q=${encodeURIComponent(q)}`);
        res.json(await r.json());
    } catch (e) { next(e); }
});

router.get("/similar", async (req, res, next) => {
    try {
        const { title, k = 10 } = req.query;
        if (!title) return res.status(400).json({ error: "title required" });

        const r = await fetch(`${CHROMA_URL}/similar/by-title?title=${encodeURIComponent(title)}&k=${k}`);
        const data = await r.json();
        res.json(data);
    } catch (e) { next(e); }
});



router.post("/similar", async (req, res, next) => {
    try {
        const r = await fetch(`${CHROMA_URL}/similar/by-embedding`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(req.body || {})
        });
        const data = await r.json();
        res.json(data);
    } catch (e) { next(e); }
});

//유사곡+임베딩 내려주기
router.get("/similar/suggest_em", async (req, res, next) => {
    try {
        const q = (req.query.q || "").trim();
        if (!q) return res.status(400).json({ error: "q required" });
        const r = await fetch(`${process.env.CHROMA_URL}/titles/suggest_em?q=${encodeURIComponent(q)}`);
        res.json(await r.json());
    } catch (e) { next(e); }
});



router.get("/emb", async (req, res, next) => {
    try {
        const q = (req.query.q || "").trim();   // 여기엔 sha256이 온다고 가정
        if (!q) {
            return res.status(400).json({ error: "q (sha256) required" });
        }

        const base = (process.env.CHROMA_URL || "").replace(/\/+$/, ""); // 끝 슬래시 제거
        const url = `${base}/emb/${encodeURIComponent(q)}`;

        const r = await fetch(url);

        if (!r.ok) {
            // FastAPI 쪽 에러 그대로 전달
            const body = await r.text().catch(() => "");
            return res
                .status(r.status)
                .json({ error: `chroma /emb 호출 실패`, status: r.status, body });
        }

        const data = await r.json();
        return res.json(data);
    } catch (e) {
        next(e);
    }
});


module.exports = router;
