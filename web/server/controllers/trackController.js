const { db } = require("../config/db");

function getPaging(req) {
    const page = Math.max(1, Number(req.query.page || 1));
    const limit = Math.min(100, Math.max(1, Number(req.query.limit || 20)));
    const offset = (page - 1) * limit;
    return { page, limit, offset };
}

async function listTracks(req, res, next) {  // GET  /api/tracks,
    try {
        const { page, limit, offset } = getPaging(req);

        const [cnt] = await db.execute("SELECT COUNT(*) AS total FROM track_embedding_meta");
        const total = cnt[0].total;

        const [rows] = await db.execute(
            `SELECT id, title, embedding_path, dim, dtype, sha256, created_at
       FROM track_embedding_meta
       ORDER BY id ASC
       LIMIT ${limit} OFFSET ${offset}`, [limit, offset]
        );

        res.json({
            total,
            totalPages: Math.ceil(total / limit),
            page,
            limit,
            items: rows
        });
    } catch (err) {
        next(err);
    }
}

async function searchTracks(req, res, next) {  //GET /api/search
    try {
        const { page, limit, offset } = getPaging(req);
        const q = (req.query.q || "").trim();
        if (!q) return res.json({ total: 0, totalPages: 0, page, limit, items: [] });

        const [cnt] = await db.execute(
            `SELECT COUNT(*) AS total FROM track_embedding_meta WHERE title LIKE ?`,
            [`%${q}%`]
        );
        const total = cnt[0].total;

        const [rows] = await db.execute(
            `SELECT id, title, embedding_path, dim, dtype, sha256, created_at
       FROM track_embedding_meta
       WHERE title LIKE ?
       ORDER BY id ASC
       LIMIT ${limit} OFFSET ${offset}`,
            [`%${q}%`, limit, offset]
        );

        res.json({
            total,
            totalPages: Math.ceil(total / limit),
            page,
            limit,
            items: rows
        });
    } catch (err) {
        next(err);
    }
}

module.exports = { listTracks, searchTracks };
