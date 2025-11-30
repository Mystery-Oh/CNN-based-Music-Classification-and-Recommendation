// import { useState } from "react"
// import BackgroundArt from "../components/BackgroundArt.jsx"
// import NeonBackground from "../components/NeonBackground.jsx";
//
// export default function SearchPage({ onPlay, setQueue }) {
//     const [q, setQ] = useState("")
//
//     // 데모 데이터 (검색 API 붙이면 교체)
//     const mock = [
//         { id: "1", title: "Remedy", artist: "Annie Schindel", albumArt: "https://picsum.photos/seed/a/600/600" },
//         { id: "2", title: "Giant",  artist: "Yuqì",            albumArt: "https://picsum.photos/seed/b/600/600" },
//         { id: "3", title: "Faith",  artist: "Nurko",           albumArt: "https://picsum.photos/seed/c/600/600" },
//     ]
//
//     const onSearch = (e) => {
//         e.preventDefault()
//         const list = mock.filter(x => (x.title + x.artist).toLowerCase().includes(q.toLowerCase()))
//         setQueue(list)
//         if (list[0]) onPlay(list[0])
//     }
//
//     return (
//         <>
//             {/*<BackgroundArt src={"https://picsum.photos/id/53/1200/800"} />*/}
//             <NeonBackground/>
//             {/* 화면 정중앙에 한 덩어리로 배치 */}
//             <div className="fixed inset-0 grid place-items-center px-6">
//                 <div className="glass w-full max-w-3xl p-8 md:p-10 text-center space-y-7">
//                     {/* 로고 칩을 카드 안으로 이동 */}
//                     <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl
//                           bg-white/15 backdrop-blur border border-white/25 text-white/90">
//                         <span className="h-4 w-4 rounded-lg bg-white/80"></span>
//                         {/*<span className="font-semibold tracking-tight">Muse UI</span>*/}
//                     </div>
//
//                     <div className="search_title text-4xl md:text-5xl font-semibold text-white/95 tracking-tight">
//                         지금 어떤 음악을 듣고 싶으신가요?
//                     </div>
//
//                     {/* 검색바 */}
//                     <form onSubmit={onSearch}
//                           className="glass w-full flex flex-col sm:flex-row items-stretch sm:items-center
//                            gap-3 px-4 py-3 text-left">
//                         <div className="flex items-center gap-3 flex-1">
//                             <span className="text-white/70 text-xl">＋</span>
//                             <input
//                                 className="bg-transparent outline-none text-lg text-white/90 placeholder:text-white/60 w-full"
//                                 placeholder="노래, 가수, 분위기…"
//                                 value={q} onChange={e=>setQ(e.target.value)}
//                             />
//                         </div>
//                         <button
//                             className="px-5 py-2 rounded-xl bg-white/85 hover:bg-white text-zinc-900 font-medium self-end sm:self-auto">
//                             검색
//                         </button>
//                     </form>
//                 </div>
//             </div>
//         </>
//     )
// }
//
//
//
// // import NeonBackground from "../components/NeonBackground.jsx";
// //
// // export default function SearchPage({ onSearch }) {
// //     const submit = (e) => { e.preventDefault(); onSearch?.(); };
// //
// //     return (
// //         <>
// //             <NeonBackground />
// //
// //             {/* 화면 정중앙 */}
// //             <div className="fixed inset-0 grid place-items-center px-6">
// //                 {/*  카드 */}
// //                 <div className="glass-pro w-full max-w-[880px] p-8 md:p-10 text-center text-white/90">
// //                     {/* 상단 작은 칩 */}
// //                     <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl
// //                           bg-white/18 backdrop-blur-md border border-white/30 text-white/90">
// //                         <span className="h-4 w-4 rounded-md bg-white/80" />
// //                         <span className="font-medium">Muse UI</span>
// //                     </div>
// //
// //                     {/* 타이틀 */}
// //                     <h1 className="mt-5 text-4xl md:text-6xl font-semibold tracking-tight drop-shadow-[0_6px_30px_rgba(0,0,0,.45)]">
// //                         지금 무슨 생각을 하시나요?
// //                     </h1>
// //
// //                     {/* 검색 바 */}
// //                     <form onSubmit={submit} className="mt-8 md:mt-10 mx-auto max-w-[720px]">
// //                         <div className="input-glass flex items-center gap-3 px-4 py-3 relative">
// //                             {/* 노이즈 얹기 */}
// //                             <div className="noise"></div>
// //
// //                             <span className="text-white/70 text-xl select-none">＋</span>
// //                             <input
// //                                 className="bg-transparent outline-none text-lg md:text-xl text-white/90
// //                            placeholder:text-white/60 w-full"
// //                                 placeholder="무엇이든 물어보세요 / 노래, 가수, 분위기…"
// //                             />
// //                             <button
// //                                 className="px-5 py-2 rounded-xl bg-white/85 hover:bg-white text-zinc-900 font-medium self-end sm:self-auto">
// //                                 검색
// //                             </button>
// //                         </div>
// //                     </form>
// //                 </div>
// //             </div>
// //         </>
// //     );
// // }


// import { useState } from "react"
// import BackgroundArt from "../components/BackgroundArt.jsx"
// import NeonBackground from "../components/NeonBackground.jsx";
//
// // onPlay(track), setQueue(list) 는 그대로 사용
// export default function SearchPage({ onPlay, setQueue }) {
//     const [q, setQ] = useState("")
//     const [loading, setLoading] = useState(false)
//     const [error, setError] = useState("")
//
//     const onSearch = async (e) => {
//         e.preventDefault()
//         const query = q.trim()
//         if (!query) return
//
//         setLoading(true)
//         setError("")
//         setQueue([])
//
//         try {
//             // ✅ Node 서버(4000) 검색 API 호출
//             // 필요에 따라 엔드포인트/파라미터는 프로젝트에 맞게 조정해줘
//             const params = new URLSearchParams({
//                 query,          // 검색어
//                 page: "1",
//                 limit: "20",
//             })
//
//             const res = await fetch(`/api/tracks?${params.toString()}`)
//             if (!res.ok) {
//                 throw new Error("검색 요청에 실패했어요.")
//             }
//
//             const data = await res.json()
//
//             // ✅ 응답 형태에 따라 유연하게 파싱
//             // 예시 가정:
//             // { items: [{ sha256, title, artist, album_art_url, ... }], ... }
//             const rawList =
//                 data.items ||
//                 data.rows ||
//                 data.results ||
//                 data.tracks ||
//                 []
//
//             // 플레이어에서 쓰기 편한 형태로 매핑
//             const list = rawList.map((x, idx) => ({
//                 id: x.id || x.sha256 || x.track_id || `${idx}`,
//                 title: x.title || x.track_title || "제목 없음",
//                 artist: x.artist || x.artist_name || x.singer || "",
//                 albumArt: x.albumArt || x.album_art_url || x.cover_url || "",
//                 // 필요하면 여기 duration, sha256 등 더 붙여도 됨
//                 _raw: x,
//             }))
//
//             setQueue(list)
//             if (list[0]) onPlay(list[0])
//             if (list.length === 0) {
//                 setError("검색 결과가 없습니다.")
//             }
//         } catch (err) {
//             console.error(err)
//             setError(err.message || "알 수 없는 오류가 발생했어요.")
//         } finally {
//             setLoading(false)
//         }
//     }
//
//     return (
//         <>
//             {/*<BackgroundArt src={"https://picsum.photos/id/53/1200/800"} />*/}
//             <NeonBackground/>
//             {/* 화면 정중앙에 한 덩어리로 배치 */}
//             <div className="fixed inset-0 grid place-items-center px-6">
//                 <div className="glass w-full max-w-3xl p-8 md:p-10 text-center space-y-7">
//                     {/* 로고 칩을 카드 안으로 이동 */}
//                     <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl
//                           bg-white/15 backdrop-blur border border-white/25 text-white/90">
//                         <span className="h-4 w-4 rounded-lg bg-white/80"></span>
//                         {/*<span className="font-semibold tracking-tight">Muse UI</span>*/}
//                     </div>
//
//                     <div className="search_title text-4xl md:text-5xl font-semibold text-white/95 tracking-tight">
//                         지금 어떤 음악을 듣고 싶으신가요?
//                     </div>
//
//                     {/* 검색바 */}
//                     <form
//                         onSubmit={onSearch}
//                         className="glass w-full flex flex-col sm:flex-row items-stretch sm:items-center
//                            gap-3 px-4 py-3 text-left"
//                     >
//                         <div className="flex items-center gap-3 flex-1">
//                             <span className="text-white/70 text-xl">＋</span>
//                             <input
//                                 className="bg-transparent outline-none text-lg text-white/90 placeholder:text-white/60 w-full"
//                                 placeholder="노래, 가수, 분위기…"
//                                 value={q}
//                                 onChange={e => setQ(e.target.value)}
//                             />
//                         </div>
//                         <button
//                             type="submit"
//                             disabled={loading}
//                             className="px-5 py-2 rounded-xl bg-white/85 hover:bg-white text-zinc-900 font-medium self-end sm:self-auto disabled:opacity-60 disabled:cursor-not-allowed"
//                         >
//                             {loading ? "검색 중…" : "검색"}
//                         </button>
//                     </form>
//
//                     {/* 에러 메시지 */}
//                     {error && (
//                         <p className="text-sm text-red-200 mt-2">
//                             {error}
//                         </p>
//                     )}
//                 </div>
//             </div>
//         </>
//     )
// }

//
// import { useState } from "react"
// import NeonBackground from "../components/NeonBackground.jsx";
//
// export default function SearchPage({ onPlay, setQueue }) {
//     const [q, setQ] = useState("")
//     const [loading, setLoading] = useState(false)
//
//     const onSearch = async (e) => {
//         e.preventDefault()
//         const query = q.trim()
//         if (!query) return
//
//         setLoading(true)
//         setQueue([])
//         setError("")
//
//         try {
//             const res = await fetch(
//                 `/api/similar?title=${encodeURIComponent(query)}&k=10`
//             )
//             const data = await res.json()
//             console.log("similar response:", data)
//
//             // 1) 제목이 없어서 detail이 온 경우
//             if (data.detail) {
//                 // 선택: 여기서 추천 제목을 UI에 보여주거나, input에 자동 채워 넣을 수도 있음
//                 const msg = data.suggest && data.suggest.length
//                     ? `${data.detail}\n\n혹시 이 곡인가요?\n- ${data.suggest.join("\n- ")}`
//                     : data.detail
//
//                 alert(msg)
//                 return
//             }
//
//             // 2) 정상 결과인 경우 (data.results 또는 data 자체가 배열일 수 있으니 방어적으로)
//             const rawList = Array.isArray(data.results)
//                 ? data.results
//                 : Array.isArray(data)
//                     ? data
//                     : []
//
//             const list = rawList.map((x, idx) => ({
//                 id: x.id || x.sha256 || `${idx}`,
//                 title: x.title || "제목 없음",
//                 artist: x.artist || "Unknown",
//                 albumArt: x.albumArt || "https://picsum.photos/600",
//                 _raw: x,
//             }))
//
//             setQueue(list)
//             if (list[0]) onPlay(list[0])
//             if (!list.length) {
//                 setError("검색 결과가 없습니다.")
//             }
//         } catch (err) {
//             console.error(err)
//             alert("검색 중 오류가 발생했습니다.")
//         } finally {
//             setLoading(false)
//         }
//     }
//
//
//     return (
//         <>
//             <NeonBackground/>
//
//             <div className="fixed inset-0 grid place-items-center px-6">
//                 <div className="glass w-full max-w-3xl p-8 md:p-10 text-center space-y-7">
//
//                     <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl
//                           bg-white/15 backdrop-blur border border-white/25 text-white/90">
//                         <span className="h-4 w-4 rounded-lg bg-white/80"></span>
//                     </div>
//
//                     <div className="search_title text-4xl md:text-5xl font-semibold text-white/95 tracking-tight">
//                         지금 어떤 음악을 듣고 싶으신가요?
//                     </div>
//
//                     <form onSubmit={onSearch}
//                           className="glass w-full flex flex-col sm:flex-row items-stretch sm:items-center
//                            gap-3 px-4 py-3 text-left">
//                         <div className="flex items-center gap-3 flex-1">
//                             <span className="text-white/70 text-xl">＋</span>
//                             <input
//                                 className="bg-transparent outline-none text-lg text-white/90 placeholder:text-white/60 w-full"
//                                 placeholder="노래, 가수, 분위기…"
//                                 value={q} onChange={e=>setQ(e.target.value)}
//                             />
//                         </div>
//                         <button
//                             disabled={loading}
//                             className="px-5 py-2 rounded-xl bg-white/85 hover:bg-white text-zinc-900 font-medium self-end sm:self-auto">
//                             {loading ? "검색 중…" : "검색"}
//                         </button>
//                     </form>
//                 </div>
//             </div>
//         </>
//     )
// }


//
// // src/pages/SearchPage.jsx
// import { useState } from "react";
// import NeonBackground from "../components/NeonBackground.jsx";
// import { fetchChromaSimilarity } from "../api/musicApi.js";
//
// export default function SearchPage({ onPlay, setQueue }) {
//     const [q, setQ] = useState("");
//     const [loading, setLoading] = useState(false);
//     const [error, setError] = useState("");
//
//     const onSearch = async (e) => {
//         e.preventDefault();
//         const query = q.trim();
//         if (!query) return;
//
//         setLoading(true);
//         if (typeof setQueue === "function") setQueue([]);
//         setError("");
//
//
//         try {
//             // ✅ 직접 fetch 대신 musicApi 사용
//             const data = await fetchChromaSimilarity(query, 10);
//             console.log("similar response:", data);
//
//             // 1) 제목이 없어서 detail이 온 경우
//             if (data.detail) {
//                 const msg =
//                     data.suggest && data.suggest.length
//                         ? `${data.detail}\n\n혹시 이 곡인가요?\n- ${data.suggest.join("\n- ")}`
//                         : data.detail;
//
//                 alert(msg);
//                 return;
//             }
//
//             // 2) 정상 결과인 경우 (data.results 또는 data 자체가 배열일 수 있으니 방어적으로)
//             const rawList = Array.isArray(data.results)
//                 ? data.results
//                 : Array.isArray(data)
//                     ? data
//                     : [];
//
//             const list = rawList.map((x, idx) => ({
//                 id: x.id || x.sha256 || `${idx}`,
//                 title: x.title || "제목 없음",
//                 artist: x.artist || "Unknown",
//                 albumArt: x.albumArt || "https://picsum.photos/600",
//                 _raw: x,
//             }));
//
//             setQueue(list);
//             if (list[0]) onPlay(list[0]);
//             if (!list.length) {
//                 setError("검색 결과가 없습니다.");
//             }
//         } catch (err) {
//             console.error(err);
//             alert("검색 중 오류가 발생했습니다.");
//         } finally {
//             setLoading(false);
//         }
//     };
//
//     return (
//         <>
//             <NeonBackground />
//
//             <div className="fixed inset-0 grid place-items-center px-6">
//                 <div className="glass w-full max-w-3xl p-8 md:p-10 text-center space-y-7">
//                     <div
//                         className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl
//                           bg-white/15 backdrop-blur border border-white/25 text-white/90"
//                     >
//                         <span className="h-4 w-4 rounded-lg bg-white/80"></span>
//                     </div>
//
//                     <div className="search_title text-4xl md:text-5xl font-semibold text-white/95 tracking-tight">
//                         지금 어떤 음악을 듣고 싶으신가요?
//                     </div>
//
//                     <form
//                         onSubmit={onSearch}
//                         className="glass w-full flex flex-col sm:flex-row items-stretch sm:items-center
//                            gap-3 px-4 py-3 text-left"
//                     >
//                         <div className="flex items-center gap-3 flex-1">
//                             <span className="text-white/70 text-xl">＋</span>
//                             <input
//                                 className="bg-transparent outline-none text-lg text-white/90 placeholder:text-white/60 w-full"
//                                 placeholder="노래, 가수, 분위기…"
//                                 value={q}
//                                 onChange={(e) => setQ(e.target.value)}
//                             />
//                         </div>
//                         <button
//                             disabled={loading}
//                             className="px-5 py-2 rounded-xl bg-white/85 hover:bg-white text-zinc-900 font-medium self-end sm:self-auto"
//                         >
//                             {loading ? "검색 중…" : "검색"}
//                         </button>
//                     </form>
//
//                     {/* 필요하면 에러 메시지도 아래에 표시 */}
//                     {error && <div className="mt-2 text-sm text-red-300">{error}</div>}
//                 </div>
//             </div>
//         </>
//     );
// }

//버전 5..
// src/pages/SearchPage.jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import NeonBackground from "../components/NeonBackground.jsx";
import { fetchChromaSimilarity } from "../api/musicApi.js";

export default function SearchPage({ onPlay, setQueue }) {
    const [q, setQ] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [suggestions, setSuggestions] = useState([]); // 유사 제목 리스트

    const navigate = useNavigate();

    // 공통 검색 함수 (input submit, 추천 클릭 둘 다 여기로)
// 공통 검색 함수 (input submit, 추천 클릭 둘 다 여기로)
    const runSearch = async (title) => {
        const query = title.trim();
        if (!query) return;

        setLoading(true);
        if (typeof setQueue === "function") setQueue([]);   // ✅ setQueue 있을 때만 초기화
        setError("");
        setSuggestions([]);

        // 공백/대소문자 무시 비교용 헬퍼
        const norm = (s) => (s || "").toLowerCase().replace(/\s+/g, "");
        try {
            const data = await fetchChromaSimilarity(query, 10);
            console.log("similar response:", data);

            // 메타데이터에 제목이 없어서 detail + suggest만 온 경우 (지금까지 쓰던 로직 유지)
            if (data.detail) {
                setError(data.detail || "");
                if (data.suggest && data.suggest.length) {
                    setSuggestions(data.suggest);
                }
                return; // 여기서는 플레이어로 안 넘어감
            }

            // 정상 결과 (유사 곡 리스트)
            const rawList = Array.isArray(data.results)
                ? data.results
                : Array.isArray(data)
                    ? data
                    : [];

            const list = rawList.map((x, idx) => ({
                id: x.id || x.sha256 || `${idx}`,
                title: x.title || "제목 없음",
                artist: x.artist || "Unknown",
                albumArt: x.albumArt || "https://picsum.photos/600",
                _raw: x,
            }));

            // 부모가 있으면 재생목록 상태 세팅
            if (typeof setQueue === "function") {
                setQueue(list);
            }

            if (!list.length) {
                setError("검색 결과가 없습니다.");
                return;
            }

            // 일치 검사
            const qNorm = norm(query);

            const exact = list.find((item) => {
                const titleNorm  = norm(item.title);
                const artistNorm = norm(item.artist);

                // 곡 제목만
                if (titleNorm === qNorm) return true;

                // "아티스트 - 제목" 형식
                const artistDashTitle = norm(`${item.artist} - ${item.title}`);
                if (artistDashTitle === qNorm) return true;

                // "제목 - 아티스트" 형식도 혹시 대비
                const titleDashArtist = norm(`${item.title} - ${item.artist}`);
                if (titleDashArtist === qNorm) return true;

                return false;
            });

            // if (exact && typeof onPlay === "function") {
            //     onPlay(exact);
            //     navigate("/player");
            //     return;
            // }


            // const exact = list.find(/* 정확 일치 찾기 */);

            // 일치 여부는 exact만 보고 판단
            if (exact) {
                console.log("정확 일치 곡 발견:", exact);

                // 재생 상태를 따로 관리하고 싶다면 있을 때만 호출
                if (typeof onPlay === "function") {
                    onPlay(exact);
                }

                // 어쨌든 플레이어 페이지 이동
                navigate("/player",{
                    state : { track : exact },
                });
                return;
            }


        } catch (err) {
            console.error(err);
            setError("검색 중 오류가 발생했습니다.");
        } finally {
            setLoading(false);
        }
    };



    // 폼 submit
    const onSearch = async (e) => {
        e.preventDefault();
        await runSearch(q);
    };

    //유튜브검색 다이렉트 테스트
    // const onSearch = (e) => {
    //     e.preventDefault();
    //     const query = q.trim();
    //     if (!query) return;
    //     navigate(`/player?q=${encodeURIComponent(query)}`);
    // };


    // 유사 제목 클릭 시
    const handleSuggestionClick = async (title) => {
        setQ(title);
        await runSearch(title);
    };

    return (
        <>
            <NeonBackground />

            <div className="fixed inset-0 grid place-items-center px-6">
                <div className="glass w-full max-w-3xl p-8 md:p-10 text-center space-y-7">
                    {/* 토글 데코 */}
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl
                          bg-white/15 backdrop-blur border border-white/25 text-white/90">
                        <span className="h-4 w-4 rounded-lg bg-white/80"></span>
                    </div>

                    {/* 타이틀 */}
                    <div className="search_title text-4xl md:text-5xl font-semibold text-white/95 tracking-tight">
                        지금 어떤 음악을 듣고 싶으신가요?
                    </div>

                    {/* 검색 폼 */}
                    <form
                        onSubmit={onSearch}
                        className="glass w-full flex flex-col sm:flex-row items-stretch sm:items-center
                       gap-3 px-4 py-3 text-left"
                    >
                        <div className="flex items-center gap-3 flex-1">
                            <span className="text-white/70 text-xl">＋</span>
                            <input
                                className="bg-transparent outline-none text-lg text-white/90 placeholder:text-white/60 w-full"
                                placeholder="노래, 가수, 분위기…"
                                value={q}
                                onChange={(e) => setQ(e.target.value)}
                            />
                        </div>
                        <button
                            disabled={loading}
                            className="px-5 py-2 rounded-xl bg-white/85 hover:bg-white text-zinc-900 font-medium self-end sm:self-auto"
                        >
                            {loading ? "검색 중…" : "검색"}
                        </button>
                    </form>

                    {/* 에러 / 안내 메시지 */}
                    {error && (
                        <div className="mt-2 text-sm text-red-200 whitespace-pre-line">
                            {error}
                        </div>
                    )}

                    {/* 유사 검색 결과 리스트 */}
                    {suggestions.length > 0 && (
                        <div className="mt-4 text-left space-y-2">
                            <div className="text-sm text-white/80">
                                혹시 이 곡인가요?
                            </div>
                            <div className="flex flex-col gap-2">
                                {suggestions.map((title, idx) => (
                                    <button
                                        key={idx}
                                        type="button"
                                        onClick={() => handleSuggestionClick(title)}
                               //          className="w-full text-left px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20
                               // text-sm text-white/90 border border-white/15 transition"
                                        className=" search_suggest_result
                                                    w-full text-left px-4 py-2 rounded-lg
                                                    bg-white/5 backdrop-blur
                                                    hover:bg-white/10 hover:scale-[1.01]
                                                    text-sm text-white/85
                                                    border border-white/10
                                                    transition-all duration-150
                                                  "
                                    >
                                        {title}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}

