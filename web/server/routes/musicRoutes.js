
const express = require("express");
const router = express.Router();
const { searchYoutubeByQuery } = require("../controllers/musicController.js")

router.get("/youtube/search", searchYoutubeByQuery);

module.exports = router;