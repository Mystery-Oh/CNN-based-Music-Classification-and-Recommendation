const fs = require("fs");
const csv = require("csv-parser");
const { db } = require("../config/db");

// 경로의 csv로 업데이트
async function updateCsv(req, res, next) {
    try {
        const csvPath = process.env.CSV_PATH || "./embeddings_v1_metadata.csv";
        if (!fs.existsSync(csvPath)) return res.status(400).json({ error: `CSV not found: ${csvPath}` });
        const result = await importCsvToDb(csvPath);
        res.json({ message: "DB 업데이트 완료", ...result });
    } catch (err) { next(err); }
}

// 파일 업로드해서 업데이트
async function updateCsvUpload(req, res, next) {
    let tmpPath;
    try {
        if (!req.file) return res.status(400).json({ error: "첨부된 CSV 파일이 없습니다. 필드명: file" });
        tmpPath = req.file.path; // multer 저장 경로
        const result = await importCsvToDb(tmpPath);
        res.json({ message: "DB 업데이트 완료", filename: req.file.originalname, ...result });
    } catch (err) {
        next(err);
    } finally {
        if (tmpPath && fs.existsSync(tmpPath)) fs.unlink(tmpPath, () => {});
    }
}

//  CSV 읽어서 DB insert/upsert

function importCsvToDb(csvPath) {
    return new Promise((resolve, reject) => {
        const sql = `
      INSERT INTO track_embedding_meta (title, embedding_path, dim, dtype, sha256)
      VALUES (?, ?, ?, ?, ?)
      ON DUPLICATE KEY UPDATE
        dim=VALUES(dim),
        dtype=VALUES(dtype),
        sha256=VALUES(sha256)
    `;

        let total = 0, valid = 0, skipped = 0, inserted = 0, updated = 0;
        const rows = [];

        //  헤더 정규화: 공백 제거 + BOM 제거 + 소문자화 >> 이거 없어서 자꾸 오류남
        const normalizer = ({ header }) =>
            header.trim().replace(/^\uFEFF/, "").toLowerCase();

        const fs = require("fs");
        const csv = require("csv-parser");

        fs.createReadStream(csvPath, { encoding: "utf8" })
            .pipe(csv({ mapHeaders: normalizer }))
            .on("data", (r) => {
                total++;

                // 컬럼명 다양한 케이스 보정 (혹시 다른 이름으로 올 때)
                const title = (r.title || r.name || "").trim();
                const embedding_path = (r.embedding_path || r.path || r.file || "").trim();
                const dtype = (r.dtype || "").trim();
                const sha256 = (r.sha256 || r.hash || "").trim();

                // dim 파싱 (숫자 아니면 무효)
                const dim = parseInt(r.dim ?? r.dimension ?? r.dimensions, 10);
                const dimOk = Number.isFinite(dim) && dim > 0;

                // 필수 값 체크 (NOT NULL 컬럼들)
                if (!title || !embedding_path || !dtype || !sha256 || !dimOk) {
                    skipped++;
                    // 디버깅 원하면 다음 줄 주석 해제:
                    // console.log("skip row:", { keys: Object.keys(r), r });
                    return;
                }

                rows.push({ title, embedding_path, dim, dtype, sha256 });
                valid++;
            })
            .on("end", async () => {
                try {
                    for (const r of rows) {
                        const params = [r.title, r.embedding_path, r.dim, r.dtype, r.sha256];
                        const [ret] = await db.execute(sql, params);
                        // 대략적인 집계
                        if (ret.affectedRows === 1 && (ret.changedRows === 0 || ret.changedRows === undefined)) {
                            inserted++;
                        } else {
                            updated++;
                        }
                    }
                    resolve({ total, valid, skipped, inserted, updated });
                } catch (e) {
                    reject(e);
                }
            })
            .on("error", reject);
    });
}




module.exports = { updateCsv, updateCsvUpload };
