require('dotenv').config()
const express = require('express')
const cors = require('cors')
const morgan = require('morgan')

const trackRoutes = require("./routes/trackRoutes");
const csvRoutes = require("./routes/csvRoutes");
const similarRoutes = require("./routes/similarRoutes");
const musicRoutes = require("./routes/musicRoutes");

const app = express()
app.use(cors())
app.use(express.json())

app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: "2mb" }));
app.use(morgan('dev'))

app.get('/', (_req, res) => res.json({ ok: true }))

app.use((err, req, res, next) => {
    console.error("서버 오류:", err);
    res.status(500).json({ error: "Internal Server Error" });
});

//api
app.use("/api", csvRoutes);    // POST /api/update-csv
app.use("/api", trackRoutes);  // GET  /api/tracks, GET /api/search

//Chroma
app.use("/api", similarRoutes);

//youtube
app.use("/api", musicRoutes)



const PORT = process.env.PORT || 4000
app.listen(PORT, () => console.log(`API on http://localhost:${PORT}`))
