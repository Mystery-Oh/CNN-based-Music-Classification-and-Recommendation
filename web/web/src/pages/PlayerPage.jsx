// import { useEffect, useRef, useState } from "react"
// import BackgroundArt from "../components/BackgroundArt.jsx"
// import Topbar from "../components/Topbar.jsx"
// import AlbumArtBlock from "../components/AlbumArtBlock.jsx"
// import Icon from "../components/Icon";
//
// export default function PlayerPage({ current, queue = [], onSelect, onQueueChange }) {
//     const audioRef = useRef(null)
//     const [progress, setProgress] = useState(0)     // 0~100
//     const [playing, setPlaying] = useState(true)
//     // 추가
//     const [time, setTime] = useState(0)
//     const [duration, setDuration] = useState(0)
//     const fmt = (s) => {
//         if (!isFinite(s)) return "0:00"
//         const m = Math.floor(s / 60)
//         const ss = Math.floor(s % 60).toString().padStart(2, "0")
//         return `${m}:${ss}`
//     }
//
//     const [likes, setLikes] = useState({})
//
//     // 현재 곡 바뀌면 자동 재생
//     useEffect(() => {
//         setProgress(0)
//         setPlaying(true)
//     }, [current?.id])
//
//     // 재생/일시정지 토글
//     useEffect(() => {
//         const a = audioRef.current
//         if (!a) return
//         if (playing) a.play().catch(() => {})      // 자동재생 차단 시 에러 무시
//         else a.pause()
//     }, [playing])
//
// // onTime 수정
//     const onTime = () => {
//         const a = audioRef.current
//         if (!a || !a.duration) return
//         setTime(a.currentTime)
//         setDuration(a.duration)
//         setProgress((a.currentTime / a.duration) * 100)
//     }
//
// // 메타 로드 시 길이 세팅
//     const onLoaded = () => {
//         const a = audioRef.current
//         if (a?.duration) setDuration(a.duration)
//     }
//
//
//     const seek = (e) => {
//         const a = audioRef.current
//         if (!a || !a.duration) return
//         const pct = Number(e.target.value)
//         a.currentTime = (pct / 100) * a.duration
//         setProgress(pct)
//     }
//
//     const next = () => {
//         if (!queue.length) return
//         const idx = queue.findIndex(t => t.id === current.id)
//         const n = queue[idx + 1] || queue[0]
//         if (n) onSelect?.(n)
//     }
//
//     const prev = () => {
//         if (!queue.length) return
//         const idx = queue.findIndex(t => t.id === current.id)
//         const p = queue[idx - 1] || queue[queue.length - 1]
//         if (p) onSelect?.(p)
//     }
//
//     // 만약 current가 없으면 빈 상태 표시
//     if (!current) {
//         return (
//             <>
//                 <BackgroundArt src={undefined} />
//                 <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
//                     <Topbar
//                         right={<span className="text-sm">재생목록22 {queue.length}곡</span>}
//                         containerClass="max-w-7xl px-4 md:px-6 lg:px-10"
//                         barClass="h-12 w-full"
//                         spacerClass="h-16"
//                     />
//                     <main className="p-4 md:p-6 lg:p-10">
//                         <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
//
//                         <div className="text-xl">재생할 곡을 선택하세요</div>
//                         </div>
//                     </main>
//                 </div>
//             </>
//         )
//     }
//
//     return (
//         <>
//             {/* 앨범 아트 블러 배경 */}
//             <BackgroundArt src={current?.albumArt} />
//
//             {/* 상단 고정 탑바 + 본문 2행 Grid 레이아웃 */}
//             <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
//                 <Topbar
//                     right={<span className="text-sm">재생목록 {queue.length}곡</span>}
//                     containerClass="max-w-[calc(100vw-64px)] px-4 md:px-6 lg:px-10"
//                     barClass="h-12 w-full"
//                     spacerClass="h-16"
//                 />
//
//                 {/* 본문: 화면을 꽉 채우는 6:4 레이아웃 */}
//                 <main className="p-4 md:p-6 lg:p-10">
//                     <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
//                         {/* 좌측(6): 현재 재생 */}
//                         {/* 좌측(6): 현재 재생 — 스크린샷 배치 */}
//                         <section className="glass h-full p-6 text-white/90 grid grid-rows-[1fr_auto] gap-6 overflow-hidden">
//                             {/* 상단: 중앙의 아트 + 제목/아티스트 오버레이 */}
//
//                             <div className="flex items-center justify-center overflow-visible">
//                                 <AlbumArtBlock
//                                     cover={current?.albumArt}
//                                     title={current?.title}
//                                     artist={current?.artist}
//                                     initialLiked={false}
//                                     onToggleLike={(v) => console.log("liked:", v)}
//                                     onMenu={() => console.log("menu")}
//                                 />
//                             </div>
//
//                             {/* 하단: 컨트롤 박스 */}
//                             <div className="rounded-2xl bg-transparent p-4">
//                                 {/* 진행바 + 시간 */}
//                                 <div className="flex items-center gap-3">
//                                     <span className="w-12 text-right tabular-nums text-white/70 text-sm">{fmt(time)}</span>
//                                     <input
//                                         type="range" min="0" max="100" value={progress} onChange={seek}
//                                         className="flex-1 accent-white/90"
//                                     />
//                                     <span className="w-12 tabular-nums text-white/70 text-sm">{fmt(duration)}</span>
//                                 </div>
//
//                                 {/* 트랜스포트 버튼 */}
//                                 <div className="mt-3 flex items-center justify-center gap-3">
//                                     {/*<button className="px-3 py-2 rounded-xl bg-white/20 hover:bg-white/30" onClick={prev} aria-label="이전">⏮</button>*/}
//                                     {/*<button className="px-5 py-2 rounded-xl font-medium bg-white text-zinc-900" onClick={() => setPlaying(p=>!p)} aria-label={playing ? "일시정지" : "재생"}>*/}
//                                     {/*    {playing ? "⏸ Pause" : "▶ Play"}*/}
//                                     {/*</button>*/}
//                                     {/*<button className="px-3 py-2 rounded-xl bg-white/20 hover:bg-white/30" onClick={next} aria-label="다음">⏭</button>*/}
//
//                                     <button
//                                         className="glass"
//                                         aria-label="이전 곡">
//                                         <Icon name="prev" size={18} className="text-white" />
//                                     </button>
//
//                                     <button
//                                         className="glass"
//                                         aria-label="재생">
//                                         <Icon name="play" size={18} className="text-white" />
//                                     </button>
//
//                                     <button
//                                         className="glass"
//                                         aria-label="다음 곡">
//                                         <Icon name="next" size={18} className="text-white" />
//                                     </button>
//
//                                 </div>
//                             </div>
//
//                             <audio
//                                 ref={audioRef}
//                                 src={current?.url || "https://www.kozco.com/tech/piano2-CoolEdit.mp3"}
//                                 onLoadedMetadata={onLoaded}
//                                 onTimeUpdate={onTime}
//                                 onEnded={next}
//                                 preload="metadata"
//                             />
//
//                         </section>
//
//
//
//                         {/* 우측(4): 재생목록 */}
//                         <aside className="glass h-full p-4 md:p-5 text-white/90 flex flex-col overflow-hidden">
//                             <h3 className="text-lg font-semibold mb-3">재생목록</h3>
//
//                             <ul className="space-y-2 overflow-y-auto pr-1 flex-1">
//                                 {queue.map(t => (
//                                     <li
//                                         key={t.id}
//                                         className={`flex items-center gap-3 p-2 rounded-xl hover:bg-white/10 cursor-pointer
//                                 ${t.id === current.id ? "bg-white/15" : ""}`}
//                                         onClick={() => onSelect?.(t)}
//                                     >
//                                         <img
//                                             src={t.albumArt}
//                                             alt=""
//                                             className="w-12 h-12 rounded-xl object-cover flex-shrink-0"
//                                         />
//                                         <div className="flex-1 min-w-0">
//                                             <div className="truncate">{t.title}</div>
//                                             <div className="text-white/60 text-sm truncate">{t.artist}</div>
//                                         </div>
//                                         <button
//                                             onClick={(e) => {
//                                                 e.stopPropagation()
//                                                 setLikes(prev => ({ ...prev, [t.id]: !prev[t.id] }))
//                                             }}
//                                             aria-label={likes[t.id] ? "좋아요 해제" : "좋아요"}
//                                             className="btn-ghost inline-flex h-9 w-9 items-center justify-center rounded-xl
//                                                        bg-transparent
//                                                        outline-none focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40">
//                                             <Icon
//                                                 name={likes[t.id] ? "heartFill" : "heart"}
//                                                 size={18}
//                                                 className={likes[t.id] ? "text-rose-500" : "text-white"}
//                                             />
//                                         </button>
//
//                                     </li>
//                                 ))}
//                             </ul>
//                         </aside>
//                     </div>
//                 </main>
//             </div>
//         </>
//     )
// }


//
// // src/pages/PlayerPage.jsx
// import { useEffect, useRef, useState, useMemo } from "react";
// import BackgroundArt from "../components/BackgroundArt.jsx";
// import Topbar from "../components/Topbar.jsx";
// import AlbumArtBlock from "../components/AlbumArtBlock.jsx";
// import Icon from "../components/Icon";
//
// export default function PlayerPage({ current, queue = [], onSelect, onQueueChange }) {
//     const audioRef = useRef(null);
//
//     // ===== 데모 데이터 (검색 API 붙이면 교체) =====
//     const mock = useMemo(() => ([
//         {
//             id: "1",
//             title: "Remedy",
//             artist: "Annie Schindel",
//             albumArt: "https://picsum.photos/seed/a/600/600",
//             url: "https://www.kozco.com/tech/piano2-CoolEdit.mp3",
//         },
//         {
//             id: "2",
//             title: "Giant",
//             artist: "Yuqì",
//             albumArt: "https://picsum.photos/seed/b/600/600",
//             url: "https://www.kozco.com/tech/piano2-CoolEdit.mp3",
//         },
//         {
//             id: "3",
//             title: "Faith",
//             artist: "Nurko",
//             albumArt: "https://picsum.photos/seed/c/600/600",
//             url: "https://www.kozco.com/tech/piano2-CoolEdit.mp3",
//         },
//     ]), []);
//
//     // ===== 내부 fallback 상태 (부모가 안 내려줄 때만 사용) =====
//     const [localQueue, setLocalQueue] = useState(queue.length ? queue : mock);
//     const [localCurrent, setLocalCurrent] = useState(current || mock[0] || null);
//
//     // 부모가 props로 내려주면 그걸 우선 사용
//     const list = (onQueueChange ? queue : localQueue) || [];
//     const cur  = (onSelect ? current : localCurrent) || list[0] || null;
//
//     // 검색 상태
//     const [q, setQ] = useState("");
//
//     // 안전하게 queue 설정(부모 or 로컬)
//     const setQueueSafe = (newQ) => {
//         if (onQueueChange) onQueueChange(newQ);
//         else setLocalQueue(newQ);
//     };
//
//     // 안전하게 재생 곡 선택(부모 or 로컬)
//     const onPlay = (track) => {
//         if (!track) return;
//         if (onSelect) onSelect(track);
//         else setLocalCurrent(track);
//     };
//
//     // 검색 실행
//     const onSearch = (e) => {
//         e.preventDefault();
//         const needle = q.trim().toLowerCase();
//         const base = mock; // 실제 API 붙이면 여기서 교체
//         const filtered = !needle
//             ? base
//             : base.filter((x) => (x.title + " " + x.artist).toLowerCase().includes(needle));
//         setQueueSafe(filtered);
//         if (filtered[0]) onPlay(filtered[0]);
//     };
//
//     // ===== 플레이어 상태 =====
//     const [progress, setProgress] = useState(0); // 0 ~ 100
//     const [playing, setPlaying] = useState(true);
//     const [time, setTime] = useState(0);
//     const [duration, setDuration] = useState(0);
//     const [likes, setLikes] = useState({});
//
//     const fmt = (s) => {
//         if (!isFinite(s)) return "0:00";
//         const m = Math.floor(s / 60);
//         const ss = Math.floor(s % 60).toString().padStart(2, "0");
//         return `${m}:${ss}`;
//     };
//
//     // 현재 곡 바뀌면 자동 재생 초기화
//     useEffect(() => {
//         setProgress(0);
//         setPlaying(true);
//     }, [cur?.id]);
//
//     // 재생/일시정지 적용
//     useEffect(() => {
//         const a = audioRef.current;
//         if (!a) return;
//         if (playing) a.play().catch(() => {}); // 자동재생 차단 시 무시
//         else a.pause();
//     }, [playing]);
//
//     // 진행 이벤트
//     const onTime = () => {
//         const a = audioRef.current;
//         if (!a || !a.duration) return;
//         setTime(a.currentTime);
//         setDuration(a.duration);
//         setProgress((a.currentTime / a.duration) * 100);
//     };
//
//     // 메타 로드 시 길이 세팅
//     const onLoaded = () => {
//         const a = audioRef.current;
//         if (a?.duration) setDuration(a.duration);
//     };
//
//     // 시크바로 탐색
//     const seek = (e) => {
//         const a = audioRef.current;
//         if (!a || !a.duration) return;
//         const pct = Number(e.target.value);
//         const t = (pct / 100) * a.duration;
//         a.currentTime = t;
//         setTime(t);
//         setDuration(a.duration);
//         setProgress(pct);
//     };
//
//     // 다음/이전
//     const next = () => {
//         if (!list.length || !cur) return;
//         const idx = list.findIndex((t) => t.id === cur.id);
//         const n = list[idx + 1] || list[0];
//         onPlay(n);
//     };
//
//     const prev = () => {
//         if (!list.length || !cur) return;
//         const idx = list.findIndex((t) => t.id === cur.id);
//         const p = list[idx - 1] || list[list.length - 1];
//         onPlay(p);
//     };
//
//     // 재생 토글 & 5초 탐색
//     const togglePlay = () => setPlaying((p) => !p);
//     const seekBy = (sec) => {
//         const a = audioRef.current;
//         if (!a || !a.duration) return;
//         const t = Math.max(0, Math.min(a.currentTime + sec, a.duration));
//         a.currentTime = t;
//         setTime(t);
//         setProgress((t / a.duration) * 100);
//     };
//
//     // 키보드 단축키: Space=토글, ←/→=5초
//     useEffect(() => {
//         const onKey = (e) => {
//             const tag = document.activeElement?.tagName?.toLowerCase();
//             if (tag === "input" || tag === "textarea") return;
//
//             if (e.code === "Space") {
//                 e.preventDefault();
//                 togglePlay();
//             } else if (e.code === "ArrowLeft") {
//                 e.preventDefault();
//                 seekBy(-5);
//             } else if (e.code === "ArrowRight") {
//                 e.preventDefault();
//                 seekBy(5);
//             }
//         };
//         window.addEventListener("keydown", onKey);
//         return () => window.removeEventListener("keydown", onKey);
//     }, []);
//
//     // current 미지정 시
//     if (!cur) {
//         return (
//             <>
//                 <BackgroundArt src={undefined} />
//                 <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
//                     <Topbar
//                         right={<span className="text-sm">재생목록 {list.length}곡</span>}
//                         containerClass="max-w-7xl px-4 md:px-6 lg:px-10"
//                         barClass="h-12 w-full"
//                         spacerClass="h-16"
//                     />
//                     <main className="p-4 md:p-6 lg:p-10">
//                         <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
//                             <div className="text-xl">재생할 곡을 선택하세요</div>
//                         </div>
//                     </main>
//                 </div>
//             </>
//         );
//     }
//
//     return (
//         <>
//             {/* 배경 */}
//             <BackgroundArt src={cur?.albumArt} />
//
//             {/* 상단 고정 레이아웃 */}
//             <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
//                 <Topbar
//                     right={<span className="text-sm">재생목록 {list.length}곡</span>}
//                     containerClass="max-w-[calc(100vw-64px)] px-4 md:px-6 lg:px-10"
//                     barClass="h-12 w-full"
//                     spacerClass="h-16"
//                 />
//
//                 {/* 본문 */}
//                 <main className="p-4 md:p-6 lg:p-10">
//                     <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
//                         {/* 좌측: 플레이어 */}
//                         <section className="glass h-full p-6 text-white/90 grid grid-rows-[1fr_auto] gap-6 overflow-hidden">
//                             {/* 앨범아트 + 제목/아티스트 */}
//                             <div className="flex items-center justify-center overflow-visible">
//                                 <AlbumArtBlock
//                                     cover={cur?.albumArt}
//                                     title={cur?.title}
//                                     artist={cur?.artist}
//                                     initialLiked={false}
//                                     onToggleLike={(v) => console.log("liked:", v)}
//                                     onMenu={() => console.log("menu")}
//                                 />
//                             </div>
//
//                             {/* 컨트롤 */}
//                             <div className="rounded-2xl bg-transparent p-4">
//                                 {/* 진행바 + 시간 */}
//                                 <div className="flex items-center gap-3">
//                   <span className="w-12 text-right tabular-nums text-white/70 text-sm">
//                     {fmt(time)}
//                   </span>
//                                     <input
//                                         type="range"
//                                         min="0"
//                                         max="100"
//                                         value={progress}
//                                         onChange={seek}
//                                         className="flex-1 accent-white/90"
//                                     />
//                                     <span className="w-12 tabular-nums text-white/70 text-sm">
//                     {fmt(duration)}
//                   </span>
//                                 </div>
//
//                                 {/* 트랜스포트 버튼 */}
//                                 <div className="mt-3 flex items-center justify-center gap-3">
//                                     <button
//                                         className="btn-ghost inline-flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 hover:bg-white/25 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         onClick={prev}
//                                         aria-label="이전 곡"
//                                     >
//                                         <Icon name="prev" size={18} className="text-white" />
//                                     </button>
//
//                                     <button
//                                         className="btn-ghost inline-flex h-10 w-24 items-center justify-center gap-2 rounded-xl bg-white text-zinc-900 hover:bg-white/90 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         onClick={togglePlay}
//                                         aria-label={playing ? "일시정지" : "재생"}
//                                     >
//                                         <Icon name={playing ? "pause" : "play"} size={18} className="text-zinc-900" />
//                                         <span className="text-sm font-medium">{playing ? "Pause" : "Play"}</span>
//                                     </button>
//
//                                     <button
//                                         className="btn-ghost inline-flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 hover:bg-white/25 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         onClick={next}
//                                         aria-label="다음 곡"
//                                     >
//                                         <Icon name="next" size={18} className="text-white" />
//                                     </button>
//                                 </div>
//                             </div>
//
//                             {/* 오디오 엘리먼트 */}
//                             <audio
//                                 ref={audioRef}
//                                 src={cur?.url || "https://www.kozco.com/tech/piano2-CoolEdit.mp3"}
//                                 onLoadedMetadata={onLoaded}
//                                 onTimeUpdate={onTime}
//                                 onEnded={next}
//                                 preload="metadata"
//                             />
//                         </section>
//
//                         {/* 우측: 검색 + 재생목록 */}
//                         <aside className="glass h-full p-4 md:p-5 text-white/90 flex flex-col overflow-hidden">
//                             {/* 검색 폼 */}
//                             <form onSubmit={onSearch} className="flex items-center gap-2 mb-3">
//                                 <input
//                                     value={q}
//                                     onChange={(e) => setQ(e.target.value)}
//                                     placeholder="제목 또는 아티스트 검색"
//                                     className="flex-1 px-3 py-2 rounded-xl bg-white/15 placeholder-white/60 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                 />
//                                 <button
//                                     type="submit"
//                                     className="px-3 py-2 rounded-xl bg-white text-zinc-900 font-medium hover:bg-white/90 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                 >
//                                     검색
//                                 </button>
//                             </form>
//
//                             <h3 className="text-lg font-semibold mb-2">재생목록</h3>
//
//                             <ul className="space-y-2 overflow-y-auto pr-1 flex-1">
//                                 {list.map((t) => (
//                                     <li
//                                         key={t.id}
//                                         className={`flex items-center gap-3 p-2 rounded-xl hover:bg-white/10 cursor-pointer ${
//                                             t.id === cur.id ? "bg-white/15" : ""
//                                         }`}
//                                         onClick={() => onPlay(t)}
//                                     >
//                                         <img
//                                             src={t.albumArt}
//                                             alt=""
//                                             className="w-12 h-12 rounded-xl object-cover flex-shrink-0"
//                                         />
//                                         <div className="flex-1 min-w-0">
//                                             <div className="truncate">{t.title}</div>
//                                             <div className="text-white/60 text-sm truncate">{t.artist}</div>
//                                         </div>
//
//                                         <button
//                                             onClick={(e) => {
//                                                 e.stopPropagation();
//                                                 setLikes((prev) => ({ ...prev, [t.id]: !prev[t.id] }));
//                                             }}
//                                             aria-label={likes[t.id] ? "좋아요 해제" : "좋아요"}
//                                             className="btn-ghost inline-flex h-9 w-9 items-center justify-center rounded-xl bg-transparent outline-none focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         >
//                                             <Icon
//                                                 name={likes[t.id] ? "heartFill" : "heart"}
//                                                 size={18}
//                                                 className={likes[t.id] ? "text-rose-500" : "text-white"}
//                                             />
//                                         </button>
//                                     </li>
//                                 ))}
//                             </ul>
//                         </aside>
//                     </div>
//                 </main>
//             </div>
//         </>
//     );
// }

//
// // src/pages/PlayerPage.jsx
// import { useEffect, useState, useMemo } from "react";
// import BackgroundArt from "../components/BackgroundArt.jsx";
// import Topbar from "../components/Topbar.jsx";
// import AlbumArtBlock from "../components/AlbumArtBlock.jsx";
// import Icon from "../components/Icon";
// import { fetchYoutubeByQuery } from "../api/musicApi.js";
//
// export default function PlayerPage({ current, queue = [], onSelect, onQueueChange }) {
//     // ===== 데모 데이터 (부모가 아무것도 안 내려줄 때용) =====
//     const mock = useMemo(
//         () => [
//             {
//                 id: "1",
//                 title: "Remedy",
//                 artist: "Annie Schindel",
//                 albumArt: "https://picsum.photos/seed/a/600/600",
//             },
//             {
//                 id: "2",
//                 title: "Giant",
//                 artist: "Yuqì",
//                 albumArt: "https://picsum.photos/seed/b/600/600",
//             },
//             {
//                 id: "3",
//                 title: "Faith",
//                 artist: "Nurko",
//                 albumArt: "https://picsum.photos/seed/c/600/600",
//             },
//         ],
//         []
//     );
//
//     // ===== 내부 fallback 상태 (부모가 안 내려줄 때만 사용) =====
//     const [localQueue, setLocalQueue] = useState(queue.length ? queue : mock);
//     const [localCurrent, setLocalCurrent] = useState(current || mock[0] || null);
//
//     // 부모가 props로 내려주면 그걸 우선 사용
//     const list = (onQueueChange ? queue : localQueue) || [];
//     const cur = (onSelect ? current : localCurrent) || list[0] || null;
//
//     // 검색 상태 (우측 재생목록 내부 검색)
//     const [q, setQ] = useState("");
//
//     // 안전하게 queue 설정(부모 or 로컬)
//     const setQueueSafe = (newQ) => {
//         if (onQueueChange) onQueueChange(newQ);
//         else setLocalQueue(newQ);
//     };
//
//     // 안전하게 재생 곡 선택(부모 or 로컬)
//     const onPlay = (track) => {
//         if (!track) return;
//         if (onSelect) onSelect(track);
//         else setLocalCurrent(track);
//     };
//
//     // 재생목록 검색 (지금은 mock 기준, 나중에 API로 교체 가능)
//     const onSearch = (e) => {
//         e.preventDefault();
//         const needle = q.trim().toLowerCase();
//         const base = mock;
//         const filtered = !needle
//             ? base
//             : base.filter((x) =>
//                 (x.title + " " + x.artist).toLowerCase().includes(needle)
//             );
//         setQueueSafe(filtered);
//         if (filtered[0]) onPlay(filtered[0]);
//     };
//
//     // ===== YouTube 검색 & 재생 상태 =====
//     const [yt, setYt] = useState(null);          // { videoId, title, thumbnail, ... }
//     const [ytLoading, setYtLoading] = useState(false);
//     const [ytError, setYtError] = useState("");
//
//     // 현재 곡이 바뀔 때마다 "제목 + 아티스트"로 유튜브 검색
//     useEffect(() => {
//         if (!cur) return;
//
//         const query = `${cur.title || ""} ${cur.artist || ""}`.trim();
//         if (!query) {
//             setYt(null);
//             setYtError("유튜브 검색에 사용할 제목/아티스트 정보가 없습니다.");
//             return;
//         }
//
//         let cancelled = false;
//
//         const run = async () => {
//             try {
//                 setYtLoading(true);
//                 setYtError("");
//                 setYt(null);
//
//                 const data = await fetchYoutubeByQuery(query);
//                 if (cancelled) return;
//
//                 if (!data || !data.videoId) {
//                     setYtError("해당 곡에 대한 유튜브 영상을 찾지 못했습니다.");
//                     setYt(null);
//                     return;
//                 }
//
//                 setYt(data);
//             } catch (err) {
//                 console.error(err);
//                 if (!cancelled) {
//                     setYtError("유튜브 검색 중 오류가 발생했습니다.");
//                 }
//             } finally {
//                 if (!cancelled) setYtLoading(false);
//             }
//         };
//
//         run();
//
//         return () => {
//             cancelled = true;
//         };
//     }, [cur?.id, cur?.title, cur?.artist]);
//
//     // 다음/이전
//     const next = () => {
//         if (!list.length || !cur) return;
//         const idx = list.findIndex((t) => t.id === cur.id);
//         const n = list[idx + 1] || list[0];
//         onPlay(n);
//     };
//
//     const prev = () => {
//         if (!list.length || !cur) return;
//         const idx = list.findIndex((t) => t.id === cur.id);
//         const p = list[idx - 1] || list[list.length - 1];
//         onPlay(p);
//     };
//
//     // 좋아요 상태
//     const [likes, setLikes] = useState({});
//
//     // current 미지정 시
//     if (!cur) {
//         return (
//             <>
//                 <BackgroundArt src={undefined} />
//                 <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
//                     <Topbar
//                         right={<span className="text-sm">재생목록 {list.length}곡</span>}
//                         containerClass="max-w-7xl px-4 md:px-6 lg:px-10"
//                         barClass="h-12 w-full"
//                         spacerClass="h-16"
//                     />
//                     <main className="p-4 md:p-6 lg:p-10">
//                         <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
//                             <div className="text-xl">재생할 곡을 선택하세요</div>
//                         </div>
//                     </main>
//                 </div>
//             </>
//         );
//     }
//
//     return (
//         <>
//             {/* 배경 */}
//             <BackgroundArt src={cur?.albumArt} />
//
//             {/* 상단 고정 레이아웃 */}
//             <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
//                 <Topbar
//                     right={<span className="text-sm">재생목록 {list.length}곡</span>}
//                     containerClass="max-w-[calc(100vw-64px)] px-4 md:px-6 lg:px-10"
//                     barClass="h-12 w-full"
//                     spacerClass="h-16"
//                 />
//
//                 {/* 본문 */}
//                 <main className="p-4 md:p-6 lg:p-10">
//                     <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
//                         {/* 좌측: 플레이어 */}
//                         <section className="glass h-full p-6 text-white/90 grid grid-rows-[1fr_auto] gap-6 overflow-hidden">
//                             {/* 앨범아트 + 제목/아티스트 */}
//                             <div className="flex items-center justify-center overflow-visible">
//                                 <AlbumArtBlock
//                                     cover={cur?.albumArt}
//                                     title={cur?.title}
//                                     artist={cur?.artist}
//                                     initialLiked={false}
//                                     onToggleLike={(v) => console.log("liked:", v)}
//                                     onMenu={() => console.log("menu")}
//                                 />
//                             </div>
//
//                             {/* 유튜브 플레이어 & 컨트롤 */}
//                             <div className="rounded-2xl bg-transparent p-4 space-y-4">
//                                 {/* 상태 메시지 */}
//                                 {ytLoading && (
//                                     <div className="text-sm text-white/70">
//                                         유튜브에서 영상을 찾는 중…
//                                     </div>
//                                 )}
//
//                                 {!ytLoading && ytError && (
//                                     <div className="text-sm text-red-300">{ytError}</div>
//                                 )}
//
//                                 {/* 유튜브 iframe */}
//                                 {!ytLoading && yt?.videoId && (
//                                     <div className="w-full aspect-video rounded-2xl overflow-hidden bg-black/60">
//                                         <iframe
//                                             className="w-full h-full"
//                                             src={`https://www.youtube.com/embed/${yt.videoId}?autoplay=1`}
//                                             title={yt.title || cur.title}
//                                             allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
//                                             allowFullScreen
//                                         />
//                                     </div>
//                                 )}
//
//                                 {/* 트랜스포트 버튼 (다음/이전 곡 제어만) */}
//                                 <div className="mt-1 flex items-center justify-center gap-3">
//                                     <button
//                                         className="btn-ghost inline-flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 hover:bg-white/25 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         onClick={prev}
//                                         aria-label="이전 곡"
//                                     >
//                                         <Icon name="prev" size={18} className="text-white" />
//                                     </button>
//
//                                     {/* 재생/일시정지는 유튜브 플레이어의 컨트롤 사용 */}
//                                     <span className="text-xs text-white/70">
//                     재생/일시정지는 유튜브 플레이어에서 조작하세요
//                   </span>
//
//                                     <button
//                                         className="btn-ghost inline-flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 hover:bg-white/25 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         onClick={next}
//                                         aria-label="다음 곡"
//                                     >
//                                         <Icon name="next" size={18} className="text-white" />
//                                     </button>
//                                 </div>
//                             </div>
//                         </section>
//
//                         {/* 우측: 검색 + 재생목록 */}
//                         <aside className="glass h-full p-4 md:p-5 text-white/90 flex flex-col overflow-hidden">
//                             {/* 검색 폼 (내부 리스트 필터용) */}
//                             <form onSubmit={onSearch} className="flex items-center gap-2 mb-3">
//                                 <input
//                                     value={q}
//                                     onChange={(e) => setQ(e.target.value)}
//                                     placeholder="제목 또는 아티스트 검색"
//                                     className="flex-1 px-3 py-2 rounded-xl bg-white/15 placeholder-white/60 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                 />
//                                 <button
//                                     type="submit"
//                                     className="px-3 py-2 rounded-xl bg-white text-zinc-900 font-medium hover:bg-white/90 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                 >
//                                     검색
//                                 </button>
//                             </form>
//
//                             <h3 className="text-lg font-semibold mb-2">재생목록</h3>
//
//                             <ul className="space-y-2 overflow-y-auto pr-1 flex-1">
//                                 {list.map((t) => (
//                                     <li
//                                         key={t.id}
//                                         className={`flex items-center gap-3 p-2 rounded-xl hover:bg-white/10 cursor-pointer ${
//                                             t.id === cur.id ? "bg-white/15" : ""
//                                         }`}
//                                         onClick={() => onPlay(t)}
//                                     >
//                                         <img
//                                             src={t.albumArt}
//                                             alt=""
//                                             className="w-12 h-12 rounded-xl object-cover flex-shrink-0"
//                                         />
//                                         <div className="flex-1 min-w-0">
//                                             <div className="truncate">{t.title}</div>
//                                             <div className="text-white/60 text-sm truncate">
//                                                 {t.artist}
//                                             </div>
//                                         </div>
//
//                                         <button
//                                             onClick={(e) => {
//                                                 e.stopPropagation();
//                                                 setLikes((prev) => ({ ...prev, [t.id]: !prev[t.id] }));
//                                             }}
//                                             aria-label={likes[t.id] ? "좋아요 해제" : "좋아요"}
//                                             className="btn-ghost inline-flex h-9 w-9 items-center justify-center rounded-xl bg-transparent outline-none focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         >
//                                             <Icon
//                                                 name={likes[t.id] ? "heartFill" : "heart"}
//                                                 size={18}
//                                                 className={likes[t.id] ? "text-rose-500" : "text-white"}
//                                             />
//                                         </button>
//                                     </li>
//                                 ))}
//                             </ul>
//                         </aside>
//                     </div>
//                 </main>
//             </div>
//         </>
//     );
// }

//
// // src/pages/PlayerPage.jsx
// import { useEffect, useState, useMemo} from "react";
// import { useLocation } from "react-router-dom";
// import BackgroundArt from "../components/BackgroundArt.jsx";
// import Topbar from "../components/Topbar.jsx";
// import Icon from "../components/Icon";
// import { fetchYoutubeByQuery } from "../api/musicApi.js"; // 유튜브 검색 API
//
// export default function PlayerPage({ current, queue = [], onSelect, onQueueChange }) {
//     const { state } = useLocation();
//     const initialTrack = state?.track || null;  // SearchPage에서 넘긴 곡
//
//
//
//     // // ===== 데모 데이터 (부모가 아무것도 안 내려줄 때용) =====
//     const mock = useMemo(() => ([
//         {
//             id: "1",
//             title: "Remedy",
//             artist: "Annie Schindel",
//             albumArt: "https://picsum.photos/seed/a/600/600",
//         },
//         {
//             id: "2",
//             title: "Giant",
//             artist: "Yuqì",
//             albumArt: "https://picsum.photos/seed/b/600/600",
//         },
//         {
//             id: "3",
//             title: "Faith",
//             artist: "Nurko",
//             albumArt: "https://picsum.photos/seed/c/600/600",
//         },
//     ]), []);
//
//     // ===== 내부 fallback 상태 (부모가 안 내려줄 때만 사용) =====
//     const [localQueue, setLocalQueue] = useState(queue.length ? queue : mock);
//     const [localCurrent, setLocalCurrent] = useState(current || mock[0] || null);
//
//     // 부모가 props로 내려주면 그걸 우선 사용s
//     const list = (onQueueChange ? queue : localQueue) || [];
//     // const cur  = (onSelect ? current : localCurrent) || list[0] || null;
//
//     const [cur, setCur] = useState(initialTrack);
//     const [videoId, setVideoId] = useState(null);
//
//     // 우측 재생목록 필터용 검색 상태
//     const [q, setQ] = useState("");
//
//     // 안전하게 queue 설정(부모 or 로컬)
//     const setQueueSafe = (newQ) => {
//         if (onQueueChange) onQueueChange(newQ);
//         else setLocalQueue(newQ);
//     };
//
//     // 안전하게 재생 곡 선택(부모 or 로컬)
//     const onPlay = (track) => {
//         if (!track) return;
//         if (onSelect) onSelect(track);
//         else setLocalCurrent(track);
//     };
//
//     // 재생목록 검색 (지금은 mock 기준, 나중에 API로 교체 가능)
//     const onSearch = (e) => {
//         e.preventDefault();
//         const needle = q.trim().toLowerCase();
//         const base = mock;
//         const filtered = !needle
//             ? base
//             : base.filter((x) => (x.title + " " + x.artist).toLowerCase().includes(needle));
//         setQueueSafe(filtered);
//         if (filtered[0]) onPlay(filtered[0]);
//     };
//
//     // ===== YouTube 검색 & 재생 상태 =====
//     const [yt, setYt] = useState(null);          // { videoId, title, thumbnail, ... }
//     const [ytLoading, setYtLoading] = useState(false);
//     const [ytError, setYtError] = useState("");
//
//     // 현재 곡(cur)이 바뀔 때마다 "제목 + 아티스트"로 유튜브 검색
//     // cur가 바뀔 때마다 유튜브 검색
//     useEffect(() => {
//         if (!cur) return;
//
//         // "가수 + 제목" 혹은 없으면 제목만
//         const q = `${cur.artist || ""} ${cur.title || ""}`.trim();
//         if (!q) return;
//
//         const run = async () => {
//             try {
//                 setYtLoading(true);
//                 setYtError("");
//                 const data = await fetchYoutubeSearch(q);
//                 console.log("youtube search response:", data);
//                 // 백엔드 응답에 맞춰서 변경 (예: data.videoId, data.id, data.items[0].id.videoId 등)
//                 setVideoId(data.videoId);
//             } catch (err) {
//                 console.error(err);
//                 setYtError("유튜브 검색 중 오류가 발생했습니다.");
//             } finally {
//                 setYtLoading(false);
//             }
//         };
//
//         run();
//     }, [cur]);   //   바뀔 때마다 다시 실행
//
//     // 다음/이전
//     const next = () => {
//         if (!list.length || !cur) return;
//         const idx = list.findIndex((t) => t.id === cur.id);
//         const n = list[idx + 1] || list[0];
//         onPlay(n);
//     };
//
//     const prev = () => {
//         if (!list.length || !cur) return;
//         const idx = list.findIndex((t) => t.id === cur.id);
//         const p = list[idx - 1] || list[list.length - 1];
//         onPlay(p);
//     };
//
//     // 좋아요 상태
//     const [likes, setLikes] = useState({});
//
//     // current 미지정 시
//     if (!cur) {
//         return (
//             <>
//                 <BackgroundArt src={undefined} />
//                 <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
//                     <Topbar
//                         right={<span className="text-sm">재생목록 {list.length}곡</span>}
//                         containerClass="max-w-7xl px-4 md:px-6 lg:px-10"
//                         barClass="h-12 w-full"
//                         spacerClass="h-16"
//                     />
//                     <main className="p-4 md:p-6 lg:p-10">
//                         <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
//                             <div className="text-xl">재생할 곡을 선택하세요</div>
//                         </div>
//                     </main>
//                 </div>
//             </>
//         );
//     }
//
//     return (
//         <>
//             {/* 배경은 그대로 앨범아트 블러 사용 */}
//             <BackgroundArt src={cur?.albumArt} />
//
//             <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
//                 <Topbar
//                     right={<span className="text-sm">재생목록 {list.length}곡</span>}
//                     containerClass="max-w-[calc(100vw-64px)] px-4 md:px-6 lg:px-10"
//                     barClass="h-12 w-full"
//                     spacerClass="h-16"
//                 />
//
//                 <main className="p-4 md:p-6 lg:p-10">
//                     <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
//                         {/* 좌측: 플레이어 */}
//                         <section className="glass h-full p-6 text-white/90 grid grid-rows-[1fr_auto] gap-6 overflow-hidden">
//                             {/* ✅ 여기: 원래 AlbumArtBlock 있던 자리에 iframe 배치 */}
//                             <div className="flex items-center justify-center overflow-visible">
//                                 <div className="w-full max-w-2xl aspect-video rounded-3xl overflow-hidden bg-black/60 border border-white/10">
//                                     {ytLoading && (
//                                         <div className="w-full h-full flex items-center justify-center text-white/70 text-sm">
//                                             유튜브에서 영상을 찾는 중…
//                                         </div>
//                                     )}
//
//                                     {!ytLoading && ytError && (
//                                         <div className="w-full h-full flex flex-col items-center justify-center text-center px-4 text-sm text-red-200">
//                                             <div className="mb-2">{ytError}</div>
//                                             <div className="text-white/70">
//                                                 [{cur.title}] {cur.artist}
//                                             </div>
//                                         </div>
//                                     )}
//
//                                     {!ytLoading && yt?.videoId && (
//                                         <iframe
//                                             className="w-full h-full"
//                                             src={`https://www.youtube.com/embed/${yt.videoId}?autoplay=1`}
//                                             title={yt.title || cur.title}
//                                             allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
//                                             allowFullScreen
//                                         />
//                                     )}
//                                 </div>
//                             </div>
//
//                             {/* 아래: 곡 정보 + 이전/다음 버튼 (재생 컨트롤은 유튜브 플레이어 사용) */}
//                             <div className="rounded-2xl bg-transparent p-4 space-y-3">
//                                 <div className="text-center">
//                                     <div className="text-lg font-semibold">{cur.title}</div>
//                                     <div className="text-sm text-white/70 mt-0.5">{cur.artist}</div>
//                                 </div>
//
//                                 <div className="mt-1 flex items-center justify-center gap-3">
//                                     <button
//                                         className="btn-ghost inline-flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 hover:bg-white/25 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         onClick={prev}
//                                         aria-label="이전 곡"
//                                     >
//                                         <Icon name="prev" size={18} className="text-white" />
//                                     </button>
//
//                                     <span className="text-xs text-white/70">
//                                         재생 / 일시정지는 유튜브 플레이어 컨트롤을 사용하세요
//                                     </span>
//
//                                     <button
//                                         className="btn-ghost inline-flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 hover:bg-white/25 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         onClick={next}
//                                         aria-label="다음 곡"
//                                     >
//                                         <Icon name="next" size={18} className="text-white" />
//                                     </button>
//                                 </div>
//                             </div>
//                         </section>
//
//                         {/* 우측: 검색 + 재생목록 */}
//                         <aside className="glass h-full p-4 md:p-5 text-white/90 flex flex-col overflow-hidden">
//                             {/* 재생목록 필터용 검색 */}
//                             <form onSubmit={onSearch} className="flex items-center gap-2 mb-3">
//                                 <input
//                                     value={q}
//                                     onChange={(e) => setQ(e.target.value)}
//                                     placeholder="제목 또는 아티스트 검색"
//                                     className="flex-1 px-3 py-2 rounded-xl bg-white/15 placeholder-white/60 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                 />
//                                 <button
//                                     type="submit"
//                                     className="px-3 py-2 rounded-xl bg-white text-zinc-900 font-medium hover:bg-white/90 outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                 >
//                                     검색
//                                 </button>
//                             </form>
//
//                             <h3 className="text-lg font-semibold mb-2">재생목록</h3>
//
//                             <ul className="space-y-2 overflow-y-auto pr-1 flex-1">
//                                 {list.map((t) => (
//                                     <li
//                                         key={t.id}
//                                         className={`flex items-center gap-3 p-2 rounded-xl hover:bg-white/10 cursor-pointer ${
//                                             t.id === cur.id ? "bg-white/15" : ""
//                                         }`}
//                                         onClick={() => onPlay(t)}
//                                     >
//                                         <img
//                                             src={t.albumArt}
//                                             alt=""
//                                             className="w-12 h-12 rounded-xl object-cover flex-shrink-0"
//                                         />
//                                         <div className="flex-1 min-w-0">
//                                             <div className="truncate">{t.title}</div>
//                                             <div className="text-white/60 text-sm truncate">{t.artist}</div>
//                                         </div>
//
//                                         <button
//                                             onClick={(e) => {
//                                                 e.stopPropagation();
//                                                 setLikes((prev) => ({ ...prev, [t.id]: !prev[t.id] }));
//                                             }}
//                                             aria-label={likes[t.id] ? "좋아요 해제" : "좋아요"}
//                                             className="btn-ghost inline-flex h-9 w-9 items-center justify-center rounded-xl bg-transparent outline-none focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
//                                         >
//                                             <Icon
//                                                 name={likes[t.id] ? "heartFill" : "heart"}
//                                                 size={18}
//                                                 className={likes[t.id] ? "text-rose-500" : "text-white"}
//                                             />
//                                         </button>
//                                     </li>
//                                 ))}
//                             </ul>
//                         </aside>
//                     </div>
//                 </main>
//             </div>
//         </>
//     );
// }


// src/pages/PlayerPage.jsx
import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import BackgroundArt from "../components/BackgroundArt.jsx";
import Topbar from "../components/Topbar.jsx";
import Icon from "../components/Icon.jsx";
import { fetchYoutubeByQuery } from "../api/musicApi.js";

export default function PlayerPage({ current, queue = [], onSelect, onQueueChange }) {
    const { state } = useLocation();
    const routeTrack = state?.track || null;
    const routeQueue = state?.queue || [];
    const navigate = useNavigate();
    const [bgImage, setBgImage] = useState(null);

    // ===== 재생목록 & 현재 곡 상태 =====
    const [localQueue, setLocalQueue] = useState(() => {
        if (queue.length) return queue;
        if (routeQueue.length) return routeQueue;
        if (routeTrack) return [routeTrack];
        return [];
    });

    const [cur, setCur] = useState(() => {
        return current || routeTrack || localQueue[0] || null;
    });

    const [likes, setLikes] = useState({});

    // props로 queue가 넘어오는 경우 동기화
    useEffect(() => {
        if (queue.length) {
            setLocalQueue(queue);
        }
    }, [queue]);



    const list = localQueue.length ? localQueue : cur ? [cur] : [];

    const selectTrack = (track) => {
        if (!track) return;
        if (typeof onSelect === "function") {
            onSelect(track); // 부모에서 current를 관리하는 구조일 경우 대비
        }
        setCur(track);
    };

    // ===== 이전/다음 곡 =====
    const goNext = () => {
        if (!list.length || !cur) return;
        const idx = list.findIndex((t) => t.id === cur.id);
        const next = list[idx + 1] || list[0];
        selectTrack(next);
    };

    const goPrev = () => {
        if (!list.length || !cur) return;
        const idx = list.findIndex((t) => t.id === cur.id);
        const prev = list[idx - 1] || list[list.length - 1];
        selectTrack(prev);
    };

    // ===== 유튜브 검색 상태 =====
    const [ytData, setYtData] = useState(null); // { videoId, title, channelTitle, thumbnail, ... }
    const [ytLoading, setYtLoading] = useState(false);
    const [ytError, setYtError] = useState("");

    // 현재 곡이 바뀔 때마다 유튜브 검색
    useEffect(() => {
        if (!cur) return;

        const q = `${cur.artist || ""} ${cur.title || ""}`.trim() || cur.title;
        if (!q) return;

        const run = async () => {
            try {
                setYtLoading(true);
                setYtError("");
                setYtData(null);

                const data = await fetchYoutubeByQuery(q);
                console.log("youtube search response:", data);

                //⚠️ 여기서 data.videoId 는 백엔드 응답 형식에 맞게 조정
                setYtData(data);
            } catch (err) {
                console.error(err);
                setYtError("유튜브 검색 중 오류가 발생했습니다.");
            } finally {
                setYtLoading(false);
            }
        };

        run();
    }, [cur?.id]); // 현재 곡이 바뀔 때마다 재검색

    const videoId = ytData?.videoId; // 응답 구조에 맞게 조정 필요

    // current가 전혀 없는 경우
    if (!cur) {
        return (
            <>
                {/*<BackgroundArt src={undefined} />*/}
                {/*<BackgroundArt src={cur?.albumArt || ytData?.thumbnail} />*/}
                <BackgroundArt src={bgImage} />

                <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
                    <Topbar
                        right={<span className="text-sm">재생목록 {list.length}곡</span>}
                        containerClass="max-w-7xl px-4 md:px-6 lg:px-10"
                        barClass="h-12 w-full"
                        spacerClass="h-16"
                    />
                    <main className="p-4 md:p-6 lg:p-10">
                        <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
                            <div className="text-xl text-white/90">재생할 곡을 선택하세요</div>
                        </div>
                    </main>
                </div>
            </>
        );
    }

    //백그라운드 배경
    useEffect(() => {
        if (ytData?.thumbnail) {
            setBgImage(ytData.thumbnail);
        } else if (cur?.albumArt) {
            setBgImage(cur.albumArt);
        }
    }, [ytData?.thumbnail, cur?.albumArt]);

    const handleAnalyze = (track) => {
        // 나중에 분석 페이지로 이동하거나, 모달 띄워서
        // 멜스펙트로그램 / 유사 곡 / 벡터 정보 보여주는 용도
        console.log("분석 버튼 클릭:", track);
        if(!track) return;

        navigate("/analyze", {
            state:{
                track,
            },
        })
    }; //END handle ana

    return (
        <>
            {/* 배경은 앨범아트나 썸네일로 */}
            {/*<BackgroundArt src={cur?.albumArt || ytData?.thumbnail} />*/}
            <BackgroundArt src={bgImage} />

            <div className="fixed inset-0 grid grid-rows-[64px_1fr]">
                <Topbar
                    right={<span className="text-sm">재생목록 {list.length}곡</span>}
                    containerClass="max-w-[calc(100vw-64px)] px-4 md:px-6 lg:px-10"
                    barClass="h-12 w-full"
                    spacerClass="h-16"
                />

                <main className="p-4 md:p-6 lg:p-10">
                    <div className="h-full max-w-7xl mx-auto grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-[6fr_4fr]">
                        {/* ===== 좌측: YouTube 플레이어 + 정보 ===== */}
                        <section className="glass h-full p-6 text-white/90 flex flex-col gap-4 overflow-hidden">
                            {/* 곡 정보 */}
                            <div>
                                <div className="text-sm uppercase tracking-wide text-white/60 mb-1">
                                    지금 재생 중
                                </div>
                                <h2 className="text-2xl md:text-3xl font-semibold leading-tight">
                                    {cur.title}
                                </h2>
                                {/*<div className="text-white/70 mt-1">{cur.artist}</div>*/}
                            </div>

                            {/* 유튜브 플레이어 영역 */}
                            <div className="mt-2 flex-1 flex flex-col">
                                <div className="aspect-video w-full rounded-2xl overflow-hidden bg-black/60">
                                    {videoId ? (
                                        <iframe
                                            src={`https://www.youtube.com/embed/${videoId}?autoplay=1`}
                                            title={cur.title || "YouTube player"}
                                            className="w-full h-full"
                                            frameBorder="0"
                                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                            allowFullScreen
                                        />
                                    ) : (
                                        <div className="w-full h-full grid place-items-center text-white/70 text-sm">
                                            {ytLoading
                                                ? "유튜브에서 영상을 찾는 중..."
                                                : ytError || "재생할 영상을 찾을 수 없습니다."}
                                        </div>
                                    )}
                                </div>

                                {/*/!* 간단 컨트롤 (이전/다음만) *!/*/}
                                {/*<div className="mt-4 flex items-center justify-center gap-3">*/}
                                {/*    <button*/}
                                {/*        className="btn-ghost inline-flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 hover:bg-white/25 outline-none focus-visible:ring-2 focus-visible:ring-white/40"*/}
                                {/*        onClick={goPrev}*/}
                                {/*        aria-label="이전 곡"*/}
                                {/*    >*/}
                                {/*        <Icon name="prev" size={18} className="text-white" />*/}
                                {/*    </button>*/}

                                {/*    <button*/}
                                {/*        className="btn-ghost inline-flex h-10 px-4 items-center justify-center gap-2 rounded-xl bg-white text-zinc-900 hover:bg-white/90 outline-none focus-visible:ring-2 focus-visible:ring-white/40"*/}
                                {/*        onClick={goNext}*/}
                                {/*        aria-label="다음 곡"*/}
                                {/*    >*/}
                                {/*        <Icon name="next" size={18} className="text-zinc-900" />*/}
                                {/*        <span className="text-sm font-medium">다음 곡</span>*/}
                                {/*    </button>*/}
                                {/*</div>*/}
                            </div>
                        </section>

                        {/* ===== 우측: 재생목록 ===== */}
                        <aside className="glass h-full p-4 md:p-5 text-white/90 flex flex-col overflow-hidden">
                            <h3 className="text-lg font-semibold mb-3">재생목록</h3>

                            <ul className="space-y-2 overflow-y-auto pr-1 flex-1">
                                {list.map((t) => (
                                    <li
                                        key={t.id}
                                        className={`flex items-center gap-3 p-2 rounded-xl hover:bg-white/10 cursor-pointer ${
                                            t.id === cur.id ? "bg-white/15" : ""
                                        }`}
                                        onClick={() => selectTrack(t)}
                                    >
                                        <img
                                            // src={t.albumArt}
                                            src={ytData?.thumbnail}
                                            alt=""
                                            className="w-12 h-12 rounded-xl object-cover flex-shrink-0"
                                        />
                                        <div className="flex-1 min-w-0">
                                            <div className="truncate">{t.title}</div>
                                            {/*<div className="text-white/60 text-sm truncate">*/}
                                            {/*    {ytData?.artist}*/}
                                            {/*</div>*/}
                                        </div>

                                        <div className="ml-auto flex items-center gap-2">
                                            <button
                                                type="button"
                                                onClick={(e) => {
                                                    e.stopPropagation();       // 리스트 클릭과 구분
                                                    handleAnalyze(t);
                                                }}
                                                className="ana_btn px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20
                                                             text-xs font-medium text-white/90
                                                             outline-none focus-visible:ring-2 focus-visible:ring-white/40">
                                                분석
                                            </button>

                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </aside>
                    </div>
                </main>
            </div>
        </>
    );
}
