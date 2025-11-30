const YT_API_KEY = process.env.YT_API_KEY;

async function searchYoutubeByQuery(req, res) {
    try {
        const { q } = req.query;
        if (!q || !q.trim()) {
            return res.status(400).json({ message: "q(검색어)가 필요합니다." });
        }

        const url = new URL("https://www.googleapis.com/youtube/v3/search");
        url.searchParams.set("part", "snippet");
        url.searchParams.set("type", "video");
        url.searchParams.set("maxResults", "1");
        url.searchParams.set("q", q);
        url.searchParams.set("key", YT_API_KEY);

        const ytRes = await fetch(url.toString());
        if (!ytRes.ok) {
            console.error("YouTube API error", await ytRes.text());
            return res.status(500).json({ message: "YouTube API 호출 실패" });
        }

        const ytData = await ytRes.json();
        const item = ytData.items?.[0];

        if (!item) {
            return res.json({ videoId: null, items: [] });
        }

        const videoId = item.id?.videoId;
        const snippet = item.snippet;

        return res.json({
            videoId,
            title: snippet.title,
            channelTitle: snippet.channelTitle,
            thumbnail: snippet.thumbnails?.medium?.url,
        });
    } catch (err) {
        console.error(err);
        return res.status(500).json({ message: "서버 오류" });
    }
}

module.exports = { searchYoutubeByQuery };