import { useEffect, useMemo, useState } from "react";
import NeonBackground from "../components/NeonBackground.jsx";
import ImageCompare from "../components/ImageCompare.jsx";
import SpectrogramImage from "../components/SpectrogramImage.jsx";

// 장르 카드용 색 구성
const GENRES = [
    { name: "발라드", c1: "#a78bfa", c2: "#f472b6" },
    { name: "록", c1: "#60a5fa", c2: "#22d3ee" },
    { name: "힙합", c1: "#fb7185", c2: "#fbbf24" },
    { name: "알앤비", c1: "#06b6d4", c2: "#a78bfa" },
    { name: "재즈", c1: "#34d399", c2: "#60a5fa" },
    { name: "팝", c1: "#f472b6", c2: "#60a5fa" },
    { name: "클래식", c1: "#cbd5e1", c2: "#93c5fd" },
    { name: "EDM", c1: "#22d3ee", c2: "#8b5cf6" },
    { name: "포크", c1: "#f59e0b", c2: "#fb7185" },
    { name: "트로트", c1: "#f43f5e", c2: "#f59e0b" },
];




const TabBtn = ({ active, children, onClick, className = "" }) => (
    <button
        onClick={onClick}
        className={`px-3 py-2 rounded-xl border transition
      ${active
            ? "bg-white text-black border-transparent shadow"   // 눌린 상태: 배경 생김
            : "bg-transparent text-black border-black/20 hover:bg-black/5"}  // 안 눌린 상태: 글자만 검정
      ${className}`}
    >
        {children}
    </button>
);

const Badge = ({ children, className = "" }) => (
    <span
        className={`absolute left-4 top-3 rounded-md px-2 py-0.5 text-xs font-semibold
                text-white/95 shadow-sm ${className}`}
    >
    {children}
  </span>
);

const ViewCard = ({ label, badgeClass, src }) => (
    <div className="relative rounded-2xl bg-white/8 border border-white/15 overflow-hidden">
        <Badge className={badgeClass}>{label}</Badge>
        <div className="p-3">
            <img
                src={src}
                alt={`${label} spectrogram`}
                className="w-full h-[min(28vh,320px)] md:h-[min(32vh,380px)] lg:h-[min(34vh,420px)] object-contain"
                loading="eager"
            />
        </div>
    </div>
);

export default function GlassWorkspace() {
    // ───────────── manifest 로드 ─────────────
    const [songs, setSongs] = useState([]);
    useEffect(() => {
        (async () => {
            try {
                const r = await fetch("/spectrograms/manifest.json");
                const j = await r.json();
                setSongs(j?.songs || []);
            } catch {
                setSongs([]);
            }
        })();
    }, []);



    const [selectedGenre, setSelectedGenre] = useState("발라드");

    // 중앙 모드
    const [mode, setMode] = useState("compare");

    // Left / Right 상태
    const [L_song, setL_song] = useState("");
    const [L_scope, setL_scope] = useState("slices");
    const [L_section, setL_section] = useState("chorus");
    const [L_index, setL_index] = useState(0);

    const [R_song, setR_song] = useState("");
    const [R_scope, setR_scope] = useState("slices");
    const [R_section, setR_section] = useState("chorus");
    const [R_index, setR_index] = useState(1);

    // Single 모드
    const [S_song, setS_song] = useState("");
    const [S_scope, setS_scope] = useState("section");
    const [S_section, setS_section] = useState("chorus");
    const [S_index, setS_index] = useState(0);

    // 곡 객체 찾기
    const findSong = (dirOrTitle) =>
        songs.find((s) => s.dir === dirOrTitle) ||
        songs.find((s) => s.title === dirOrTitle) ||
        null;

    // manifest 로드 후 기본값 세팅
    useEffect(() => {
        if (!songs.length) return;
        const first = songs[0];
        setL_song((p) => p || first.dir);
        setR_song((p) => p || songs[1]?.dir || first.dir);
        setS_song((p) => p || first.dir);

        const defaultSection =
            Object.keys(first.sections || { chorus: [] })[0] || "chorus";
        setL_section(defaultSection);
        setR_section(defaultSection);
        setS_section(defaultSection);
    }, [songs]);

    // 스코프에 따른 파일 리스트 만들기
    const listCandidates = (songObj, scope, section) => {
        if (!songObj) return [];
        if (scope === "slices") return songObj.slices || [];
        const arr = songObj.sections?.[section] || [];
        return arr.map((name) => `${section}/${name}`); // 섹션 폴더 접두 붙이기
    };

    // 계산 값들
    const L_songObj = useMemo(() => findSong(L_song), [songs, L_song]);
    const R_songObj = useMemo(() => findSong(R_song), [songs, R_song]);
    const S_songObj = useMemo(() => findSong(S_song), [songs, S_song]);

    const L_files = useMemo(
        () => listCandidates(L_songObj, L_scope, L_section),
        [L_songObj, L_scope, L_section]
    );
    const R_files = useMemo(
        () => listCandidates(R_songObj, R_scope, R_section),
        [R_songObj, R_scope, R_section]
    );
    const S_files = useMemo(
        () => listCandidates(S_songObj, S_scope, S_section),
        [S_songObj, S_scope, S_section]
    );

    useEffect(() => {
        if (L_index >= L_files.length) setL_index(0);
    }, [L_files, L_index]);
    useEffect(() => {
        if (R_index >= R_files.length) setR_index(0);
    }, [R_files, R_index]);
    useEffect(() => {
        if (S_index >= S_files.length) setS_index(0);
    }, [S_files, S_index]);

    const L_src = useMemo(() => {
        const f = L_files[L_index];
        if (!L_songObj || !f) return "";
        return encodeURI(`${L_songObj.dir}/${f}`);
    }, [L_songObj, L_files, L_index]);

    const R_src = useMemo(() => {
        const f = R_files[R_index];
        if (!R_songObj || !f) return "";
        return encodeURI(`${R_songObj.dir}/${f}`);
    }, [R_songObj, R_files, R_index]);

    const S_src = useMemo(() => {
        const f = S_files[S_index];
        if (!S_songObj || !f) return "";
        return encodeURI(`${S_songObj.dir}/${f}`);
    }, [S_songObj, S_files, S_index]);

    const showDebug = false; // 경로 박스 숨김(원하면 true)

    useEffect(() => {
        if (mode === "single") setS_index(0);
    }, [mode]);


    return (
        // 페이지 스크롤 제거 + 안쪽 요소가 줄어들 수 있게
        <div className="fixed inset-0 overflow-hidden bg-[#0a0b13] text-white/90">
            <NeonBackground />

            {/* 3열: 좌 320, 가운데 1fr, 우 340 / min-h-0 중요 */}
            <div className="h-full max-w-[1920px] mx-auto px-5 py-5 grid grid-cols-[320px_minmax(0,1fr)_340px] gap-5 min-h-0">
                {/* 좌측: 장르 2×N 정사각형 그리드 */}
                <aside className="relative glass p-4 shadow-float glare hairline grid grid-rows-[auto_1fr] min-h-0">
                    <div className="text-sm font-semibold mb-3 opacity-90">Genres</div>

                    <div className="relative min-h-0">
                        <div className="grid grid-cols-2 gap-3">
                            {GENRES.map((g) => {
                                const active = selectedGenre === g.name;
                                return (
                                    // ✅ 정사각형 껍질: 가로폭 기준으로 높이를 100%로 맞춤 (padding-bottom: 100%)
                                    <div key={g.name} className="relative w-full pb-[90%]">
                                        {/* 껍질을 꽉 채우는 버튼 */}
                                        <button
                                            onClick={() => setSelectedGenre(g.name)}
                                            aria-pressed={active}
                                            className={`group absolute inset-0 rounded-2xl overflow-hidden transition outline-none focus:outline-none
                          ${active ? "ring-2 ring-white/70" : "ring-0"}`}
                                            style={{
                                                background: `
                  radial-gradient(120% 120% at 30% 30%, ${g.c1}99 0%, transparent 60%),
                  radial-gradient(120% 120% at 70% 70%, ${g.c2}cc 0%, transparent 55%)
                `,
                                                filter: "saturate(1.1)",
                                            }}
                                        >
                                            {/* 글래스 레이어 */}
                                            <div className="absolute inset-0 bg-white/14 backdrop-blur-xl border border-white/25 rounded-2xl" />
                                            <div className="pointer-events-none absolute inset-0 rounded-2xl shadow-[inset_0_0_0_1px_rgba(255,255,255,.06)]" />
                                            <div className="pointer-events-none absolute inset-x-0 top-0 h-[38%] rounded-t-2xl bg-gradient-to-b from-white/35 to-transparent" />

                                            {/* 라벨 */}
                                            <span className="relative z-[1] flex h-full items-center justify-center text-center
                                 px-2 text-white font-semibold text-[clamp(14px,1.6vw,18px)]
                                 drop-shadow-[0_4px_16px_rgba(0,0,0,.45)]">
                {g.name}
              </span>

                                            {/* 활성 체크 표시 */}
                                            <span className={`absolute right-2.5 top-2.5 z-[1] h-6 w-6 rounded-lg grid place-items-center
                                transition ${active ? "bg-white text-zinc-900" : "bg-white/25 text-white/90"}`}>
                ✓
              </span>
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </aside>


                {/* 가운데: 뷰어(두 장 분할), 내부가 줄어들 수 있어야 함 */}
                <main className="relative glass-deep glare hairline shadow-float p-6 grid gap-6">
                    {mode === "compare" ? (
                        <>
                            <ViewCard label="Left"  badgeClass="bg-emerald-500/90" src={L_src} />
                            <ViewCard label="Right" badgeClass="bg-sky-500/90"     src={R_src} />
                        </>
                    ) : (
                        <ViewCard label="Single" badgeClass="bg-fuchsia-500/90" src={S_src} />
                    )}
                </main>


                {/* 우측: 컴팩트 컨트롤 */}
                <aside className="glass p-4 glare hairline shadow-float flex flex-col gap-4 min-h-0">
                    {/* 모드 */}
                    <div>
                        <div className="text-sm font-semibold mb-2 opacity-90">Mode</div>
                        {/*<div className="grid grid-cols-2 gap-2">*/}
                        {/*    <button*/}
                        {/*        className={`px-3 py-1.5 rounded-xl text-sm ${*/}
                        {/*            mode === "compare"*/}
                        {/*                ? "bg-white text-zinc-900"*/}
                        {/*                : "bg-white/15 hover:bg-white/25"*/}
                        {/*        }`}*/}
                        {/*        onClick={() => setMode("compare")}*/}
                        {/*    >*/}
                        {/*        Compare*/}
                        {/*    </button>*/}
                        {/*    <button*/}
                        {/*        className={`px-3 py-1.5 rounded-xl text-sm ${*/}
                        {/*            mode === "single"*/}
                        {/*                ? "bg-white text-zinc-900"*/}
                        {/*                : "bg-white/15 hover:bg-white/25"*/}
                        {/*        }`}*/}
                        {/*        onClick={() => setMode("single")}*/}
                        {/*    >*/}
                        {/*        Single*/}
                        {/*    </button>*/}
                        {/*</div>*/}
                        <div className="grid grid-cols-2 gap-2">
                            <TabBtn active={mode === "compare"} onClick={() => setMode("compare")}>
                                Compare
                            </TabBtn>
                            <TabBtn active={mode === "single"} onClick={() => setMode("single")}>
                                Single
                            </TabBtn>
                        </div>

                    </div>

                    {/* Compare 컨트롤 */}
                    {mode === "compare" && (
                        <div className="space-y-4 overflow-auto min-h-0">
                            {/* LEFT */}
                            <div className="glass p-3 rounded-2xl space-y-2">
                                <div className="font-semibold text-sm">Left</div>
                                <select
                                    className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                    value={L_song}
                                    onChange={(e) => setL_song(e.target.value)}
                                >
                                    {songs.map((s) => (
                                        <option key={s.dir} value={s.dir}>
                                            {s.title}
                                        </option>
                                    ))}
                                </select>

                                <div className="grid grid-cols-2 gap-2">
                                    {/*<button*/}
                                    {/*    className={`px-3 py-1.5 text-sm rounded-xl ${*/}
                                    {/*        L_scope === "slices"*/}
                                    {/*            ? "bg-white text-zinc-900"*/}
                                    {/*            : "bg-white/15 hover:bg-white/25"*/}
                                    {/*    }`}*/}
                                    {/*    onClick={() => setL_scope("slices")}*/}
                                    {/*>*/}
                                    {/*    Slices*/}
                                    {/*</button>*/}
                                    {/*<button*/}
                                    {/*    className={`px-3 py-1.5 text-sm rounded-xl ${*/}
                                    {/*        L_scope === "section"*/}
                                    {/*            ? "bg-white text-zinc-900"*/}
                                    {/*            : "bg-white/15 hover:bg-white/25"*/}
                                    {/*    }`}*/}
                                    {/*    onClick={() => setL_scope("section")}*/}
                                    {/*>*/}
                                    {/*    Section*/}
                                    {/*</button>*/}

                                    <TabBtn active={L_scope === "slices"} onClick={() => setL_scope("slices")}>
                                        Slices
                                    </TabBtn>
                                    <TabBtn active={L_scope === "section"} onClick={() => setL_scope("section")}>
                                        Section
                                    </TabBtn>


                                </div>

                                {L_scope === "section" && (
                                    <select
                                        className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                        value={L_section}
                                        onChange={(e) => {
                                            setL_section(e.target.value);
                                            setL_index(0);
                                        }}
                                    >
                                        {Object.keys(L_songObj?.sections || {}).map((sec) => (
                                            <option key={sec} value={sec}>
                                                {sec}
                                            </option>
                                        ))}
                                    </select>
                                )}

                                <select
                                    className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                    value={L_index}
                                    onChange={(e) => setL_index(Number(e.target.value))}
                                >
                                    {L_files.map((f, i) => (
                                        <option key={f} value={i}>
                                            {f}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* RIGHT */}
                            <div className="glass p-3 rounded-2xl space-y-2">
                                <div className="font-semibold text-sm">Right</div>
                                <select
                                    className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                    value={R_song}
                                    onChange={(e) => setR_song(e.target.value)}
                                >
                                    {songs.map((s) => (
                                        <option key={s.dir} value={s.dir}>
                                            {s.title}
                                        </option>
                                    ))}
                                </select>

                                <div className="grid grid-cols-2 gap-2">
                                    {/*<button*/}
                                    {/*    className={`px-3 py-1.5 text-sm rounded-xl ${*/}
                                    {/*        R_scope === "slices"*/}
                                    {/*            ? "bg-white text-zinc-900"*/}
                                    {/*            : "bg-white/15 hover:bg-white/25"*/}
                                    {/*    }`}*/}
                                    {/*    onClick={() => setR_scope("slices")}*/}
                                    {/*>*/}
                                    {/*    Slices*/}
                                    {/*</button>*/}
                                    {/*<button*/}
                                    {/*    className={`px-3 py-1.5 text-sm rounded-xl ${*/}
                                    {/*        R_scope === "section"*/}
                                    {/*            ? "bg-white text-zinc-900"*/}
                                    {/*            : "bg-white/15 hover:bg-white/25"*/}
                                    {/*    }`}*/}
                                    {/*    onClick={() => setR_scope("section")}*/}
                                    {/*>*/}
                                    {/*    Section*/}
                                    {/*</button>*/}
                                    <TabBtn active={R_scope === "slices"} onClick={() => setR_scope("slices")}>
                                        Slices
                                    </TabBtn>
                                    <TabBtn active={R_scope === "section"} onClick={() => setR_scope("section")}>
                                        Section
                                    </TabBtn>
                                </div>

                                {R_scope === "section" && (
                                    <select
                                        className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                        value={R_section}
                                        onChange={(e) => {
                                            setR_section(e.target.value);
                                            setR_index(0);
                                        }}
                                    >
                                        {Object.keys(R_songObj?.sections || {}).map((sec) => (
                                            <option key={sec} value={sec}>
                                                {sec}
                                            </option>
                                        ))}
                                    </select>
                                )}

                                <select
                                    className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                    value={R_index}
                                    onChange={(e) => setR_index(Number(e.target.value))}
                                >
                                    {R_files.map((f, i) => (
                                        <option key={f} value={i}>
                                            {f}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    )}

                    {/* Single 컨트롤 */}
                    {mode === "single" && (
                        <div className="space-y-3 overflow-auto min-h-0">
                            <div className="text-sm font-semibold">Song</div>

                            <select
                                className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                value={S_song}
                                onChange={(e) => {
                                    setS_song(e.target.value);
                                    setS_index(0);               // 곡 바꾸면 첫 항목부터
                                }}
                            >
                                {songs.map((s) => (
                                    <option key={s.dir} value={s.dir}>{s.title}</option>
                                ))}
                            </select>

                            <div className="grid grid-cols-2 gap-2">
                                <TabBtn active={S_scope === "slices"} onClick={() => { setS_scope("slices"); setS_index(0); }}>
                                    Slices
                                </TabBtn>
                                <TabBtn active={S_scope === "section"} onClick={() => { setS_scope("section"); setS_index(0); }}>
                                    Section
                                </TabBtn>
                            </div>

                            {S_scope === "section" && (
                                <select
                                    className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                    value={S_section}
                                    onChange={(e) => { setS_section(e.target.value); setS_index(0); }}
                                >
                                    {Object.keys(S_songObj?.sections || {}).map((sec) => (
                                        <option key={sec} value={sec}>{sec}</option>
                                    ))}
                                </select>
                            )}

                            <select
                                className="w-full px-3 py-2 text-sm rounded-xl bg-white/15 hover:bg-white/25"
                                value={S_index}
                                onChange={(e) => setS_index(Number(e.target.value))}
                            >
                                {S_files.map((f, i) => (
                                    <option key={f} value={i}>{f}</option>
                                ))}
                            </select>
                        </div>
                    )}


                    {/* 경로 디버그 (기본 숨김) */}
                    {showDebug && (
                        <div className="glass p-3 text-xs">
                            <div className="text-white/70">현재 Left</div>
                            <div className="break-all">{L_src}</div>
                            <div className="text-white/70 mt-2">현재 Right / Single</div>
                            <div className="break-all">{mode === "compare" ? R_src : S_src}</div>
                        </div>
                    )}
                </aside>
            </div>
        </div>
    );
}
