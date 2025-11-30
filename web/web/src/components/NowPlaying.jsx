export default function NowPlaying({ track, onPrev, onNext, onToggle }) {
    if (!track) return (
        <div className="card-like h-full flex items-center justify-center text-zinc-500">
            선택된 곡이 없습니다.
        </div>
    )

    return (
        <div className="grid gap-5">
            <section className="card-like p-5">
                <div className="aspect-square w-full overflow-hidden rounded-2xl mb-4">
                    <img src={track.image} alt={track.title} className="h-full w-full object-cover" />
                </div>
                <div className="space-y-1">
                    <div className="text-xl font-semibold">{track.title}</div>
                    <div className="text-zinc-500">{track.artist}</div>
                </div>
            </section>

            <section className="card-like p-5">
                {/* 시크/컨트롤러 */}
                <div className="mb-4">
                    <div className="h-2 rounded-full bg-white/15">
                        <div className="h-2 rounded-full bg-white/80" style={{ width: '42%' }} />
                    </div>
                    <div className="mt-1 flex justify-between text-xs text-zinc-400">
                        <span>1:12</span><span>3:30</span>
                    </div>
                </div>

                <div className="flex items-center justify-center gap-4">
                    <button className="pill" onClick={onPrev}>⏮</button>
                    <button className="pill-lg" onClick={onToggle}>⏯</button>
                    <button className="pill" onClick={onNext}>⏭</button>
                </div>
            </section>
        </div>
    )
}
