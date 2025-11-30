
import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import BackgroundArt from "../components/BackgroundArt.jsx";
import Topbar from "../components/Topbar.jsx";
import { fetchChromaSimilarity } from "../api/musicApi.js";

export default function AnalyzePage() {
    const { state } = useLocation();
    const navigate = useNavigate();
    const track = state?.track || null;   // PlayerPage에서 넘겨준 곡 정보

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [neighbors, setNeighbors] = useState([]);

    // 멜/벡터는 일단 track._raw 안에 뭐가 올 수 있다고 가정하고 자리만 만들어 둠
    // 실제 필드명에 맞게 나중에 바꾸면 됨
    const melUrl =
        track?._raw?.mel_url ||
        track?._raw?.melSpectrogramUrl ||
        null;

    const vector =
        Array.isArray(track?._raw?.vector)
            ? track._raw.vector
            : Array.isArray(track?._raw?.embedding)
                ? track._raw.embedding
                : [];



    function attachSimilarity(list) {
        const scores = list.map(n => n.score);
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        const diff = max - min || 1;

        return list.map(n => {
            const sim = (max - n.score) / diff; // 0~1
            return {
                ...n,
                similarity: sim,
                similarityPct: sim * 100,
            };
        });
    }

    // 유사곡 다시 조회 (현재 곡 기준으로 k개)
    useEffect(() => {
        if (!track?.title) return;

        const run = async () => {
            try {
                setLoading(true);
                setError("");
                const data = await fetchChromaSimilarity(track.title, 10);
                console.log("analysis similar response:", data);

                const rawList = Array.isArray(data.results)
                    ? data.results
                    : Array.isArray(data)
                        ? data
                        : [];

                const mapped = rawList.map((x, idx) => ({
                    id: x.id || x.sha256 || `${idx}`,
                    title: x.title || "제목 없음",
                    artist: x.artist || "Unknown",
                    albumArt:
                        x.albumArt ||
                        track.albumArt ||
                        "https://picsum.photos/seed/analysis/400/400",
                    score:
                        typeof x.score === "number"
                            ? x.score
                            : typeof x.distance === "number"
                                ? 1 - x.distance
                                : undefined,
                    _raw: x,
                }));

                // setNeighbors(mapped);
                const withSim = attachSimilarity(mapped);
                setNeighbors(withSim);

            } catch (err) {
                console.error(err);
                setError("유사 곡 분석 중 오류가 발생했습니다.");
            } finally {
                setLoading(false);
            }
        };

        run();
    }, [track?.title, track?.albumArt]);

    // 곡 정보가 아예 없을 때
    if (!track) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-black text-white">
                <div className="text-center space-y-4">
                    <div className="text-xl">분석할 곡 정보가 없습니다.</div>
                    <button
                        className="ana_btn px-4 py-2 rounded-xl bg-white text-zinc-900"
                        onClick={() => navigate("/")}
                    >
                        검색 화면으로 돌아가기
                    </button>
                </div>
            </div>
        );
    }

    return (
        <>
            <BackgroundArt src={track.albumArt || melUrl || undefined} />

            <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
                <Topbar
                    right={
                        <button
                            className="ana_btn px-3 py-1.5 rounded-xl bg-white/15 hover:bg-white/25 text-xs text-white/90"
                            onClick={() => navigate(-1)}
                        >
                            돌아가기
                        </button>
                    }
                    containerClass="max-w-[calc(100vw-64px)] px-4 md:px-6 lg:px-10"
                    barClass="h-12 w-full"
                    spacerClass="h-16"
                />

                <main className="p-4 md:p-6 lg:p-10">
                    <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 lg:grid-cols-[6fr_4fr] text-white/90">
                        {/* ===== 좌측: 곡 정보 + 멜스펙트로그램 + 벡터 ===== */}
                        <section className="glass h-full p-6 flex flex-col gap-5 overflow-hidden">
                            {/* 곡 기본 정보 */}
                            <div className="flex items-center gap-4">
                                <img
                                    src={track.albumArt}
                                    alt=""
                                    className="w-16 h-16 rounded-xl object-cover flex-shrink-0"
                                />
                                <div className="min-w-0">
                                    <div className="text-xs uppercase tracking-[0.2em] text-white/60 mb-1">
                                        분석 중인 곡
                                    </div>
                                    <h1 className="text-xl md:text-2xl font-semibold truncate">
                                        {track.title}
                                    </h1>
                                    {/*<div className="text-white/70 truncate">{track.artist}</div>*/}
                                </div>
                            </div>

                            {/* 로딩 / 에러 */}
                            {loading && (
                                <div className="mt-2 text-sm text-white/70">
                                    유사 곡과 벡터 정보를 불러오는 중입니다…
                                </div>
                            )}
                            {error && (
                                <div className="mt-2 text-sm text-red-300 whitespace-pre-line">
                                    {error}
                                </div>
                            )}

                            {/* 멜스펙트로그램 영역 */}
                            <div className="mt-2">
                                <div className="text-sm font-medium text-white/80 mb-2">
                                    멜스펙트로그램
                                </div>
                                <div className="aspect-video rounded-2xl overflow-hidden bg-black/60 border border-white/10 flex items-center justify-center">
                                    {melUrl ? (
                                        <img
                                            src={melUrl}
                                            alt="Mel-spectrogram"
                                            className="w-full h-full object-contain"
                                        />
                                    ) : (
                                        <div className="text-xs text-white/60 px-4 text-center">
                                            멜스펙트로그램 이미지가 없어요...
                                            <br />

                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* 벡터 요약 영역 */}
                            <div className="mt-3">
                                <div className="flex items-center justify-between mb-2">
                                    <div className="text-sm font-medium text-white/80">
                                        임베딩 벡터
                                    </div>
                                    <div className="text-xs text-white/60">
                                        차원 수: {vector.length || 0}
                                    </div>
                                </div>

                                <div className="rounded-2xl bg-white/5 border border-white/10 p-3 text-xs text-white/80 max-h-32 overflow-y-auto">
                                    {vector.length ? (
                                        <div className="grid grid-cols-4 gap-2">
                                            {vector.slice(0, 32).map((v, idx) => (
                                                <div
                                                    key={idx}
                                                    className="flex items-center justify-between gap-1"
                                                >
                                                    <span className="text-white/50">d{idx}</span>
                                                    <span className="tabular-nums">
                            {v.toFixed ? v.toFixed(3) : v}
                          </span>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="text-white/60">
                                            벡터 정보가 없습니다.
                                            <br />
                                            (Chroma에서 임베딩 배열을 응답에 포함시키면 여기서 일부를
                                            미리보기로 보여줄 수 있어요??_뭔말임.)
                                        </div>
                                    )}
                                </div>
                            </div>
                        </section>

                        {/* ===== 우측: 유사 곡 리스트 ===== */}
                        <aside className="glass h-full p-6 flex flex-col overflow-hidden">
                            <div className="flex items-center justify-between mb-3">
                                <h2 className="text-lg font-semibold">유사한 곡</h2>
                                <span className="text-xs text-white/60">
                  {neighbors.length}곡
                </span>
                            </div>

                            <div className="flex-1 overflow-y-auto pr-1 space-y-2">
                                {!loading && neighbors.length === 0 && (
                                    <div className="text-sm text-white/60">
                                        아직 유사 곡 정보가 없습니다.
                                        <br />
                                    </div>
                                )}

                                {neighbors.map((n) => (
                                    <div
                                        key={n.id}
                                        className="flex items-center gap-3 p-2 rounded-xl bg-white/5 hover:bg-white/10"
                                    >
                                        <img
                                            src={n.albumArt}
                                            alt=""
                                            className="w-10 h-10 rounded-lg object-cover flex-shrink-0"
                                        />
                                        <div className="flex-1 min-w-0">
                                            <div className="truncate text-sm">{n.title}</div>
                                            {/*<div className="text-xs text-white/60 truncate">*/}
                                            {/*    {n.artist}*/}
                                            {/*</div>*/}
                                        </div>
                                        {typeof n.score === "number" && (
                                            <div className="text-xs text-white/60 tabular-nums">
                                                {/*{(n.score * 100).toFixed(1)}%*/}
                                                {n.similarityPct.toFixed(1)}%
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </aside>
                    </div>
                </main>
            </div>
        </>
    );
}
