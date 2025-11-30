const express = require("express");
const { listTracks, searchTracks } = require("../controllers/trackController");

const router = express.Router();
router.get("/tracks", listTracks);
router.get("/search", searchTracks);

module.exports = router;
