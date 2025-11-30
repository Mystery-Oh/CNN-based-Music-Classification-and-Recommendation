const express = require("express");
const multer  = require("multer");
const path    = require("path");
const fs = require("fs");
const { updateCsv, updateCsvUpload } = require("../controllers/csvController");

// // 업로드 설정: tmp 폴더 저장, 10MB 제한, CSV만 허용
// const upload = multer({
//     dest: path.join(process.cwd(), "tmp"),
//     limits: { fileSize: 10 * 1024 * 1024 },
//     fileFilter: (_req, file, cb) => {
//         const ok = ["text/csv", "application/vnd.ms-excel"].includes(file.mimetype)
//             || file.originalname.toLowerCase().endsWith(".csv");
//         cb(ok ? null : new Error("10mb 미만 CSV 파일만 업로드 가능합니다."));
//     }
// });

const tmpDir = path.join(process.cwd(), "tmp");
fs.mkdirSync(tmpDir, { recursive: true });


const storage = multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, tmpDir),
    filename: (_req, file, cb) => cb(null, Date.now() + "_" + file.originalname)
});

const upload = multer({ storage /*, fileFilter*/ });

const router = express.Router();
router.post("/update-csv", updateCsv); //POST /api/update-csv
//개별 업로드
router.post("/update-csv-upload", upload.single("file"), updateCsvUpload);

module.exports = router;
