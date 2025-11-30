import { useState } from "react"
import Icon from "../components/Icon";

export default function AlbumArtBlock({
                                          cover,           // 이미지 URL
                                          title,           // 곡 제목
                                          artist,          // 가수
                                          initialLiked,    // (옵션) 기본 좋아요
                                          onToggleLike,    // (옵션) 좋아요 토글 콜백
                                          onMenu,          // (옵션) 삼단점 메뉴 콜백
                                      }) {
    const [liked, setLiked] = useState(!!initialLiked)

    const toggle = () => {
        const next = !liked
        setLiked(next)
        onToggleLike?.(next)
    }

    return (
        <div className="relative w-[min(56vh,520px)]">
            {/* 앨범 아트 */}
            <img
                src={cover}
                alt=""
                className="w-full aspect-square object-cover rounded-2xl shadow-[0_20px_60px_-20px_rgba(0,0,0,0.45)]"
            />

            {/* 하단 오버레이: 하트 | 제목+가수 | 메뉴(삼단점) */}
            <div className="absolute left-1/2 -translate-x-1/2 bottom-3 w-[92%]">
                <div className="flex items-center gap-3 rounded-2xl bg-white/16 backdrop-blur-md border border-white/25 px-4 py-3">
                    {/* 하트 */}
                    <button
                        onClick={() => setLiked(v => !v)}
                        aria-pressed={liked}
                        aria-label={liked ? "좋아요 해제" : "좋아요"}
                        className={`btn-ghost inline-flex items-center justify-center rounded-xl p-2 transition outline-none focus:outline-none focus:ring-0 focus-visible:ring-0 ring-0";
              ${liked ? " text-rose-600" : " "}`}
                    >
                        <Icon name={liked ? "heartFill" : "heart"} size={18} />
                    </button>

                    {/* 제목 + 가수 (가운데 정렬, 줄바꿈/말줄임) */}
                    <div className="flex-1 min-w-0 text-center">
                        <div className="truncate text-[1.15rem] md:text-[1.25rem] font-semibold leading-tight">
                            {title}
                        </div>
                        <div className="truncate text-white/80 text-sm mt-0.5">
                            {artist}
                        </div>
                    </div>

                    {/* 삼단점 메뉴 */}
                    <button className="btn-ghost focus:outline-none inline-flex items-center justify-center rounded-xl bg-white/30 hover:bg-white p-2" aria-label="메뉴">
                        <Icon name="dotsVertical" size={18} className="text-white/90" />
                    </button>
                </div>
            </div>
        </div>
    )
}
