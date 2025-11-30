export default function Playlist({ items, currentId, onPick }) {
    return (
        <section className="card-like p-4 h-full overflow-hidden flex flex-col">
            <div className="px-1 pb-3 flex items-center justify-between">
                <h3 className="font-semibold">재생목록</h3>
            </div>
            <div className="flex-1 overflow-auto pr-1">
                <ul className="space-y-2">
                    {items.map((t) => {
                        const active = t.id === currentId
                        return (
                            <li key={t.id}>
                                <button
                                    onClick={() => onPick?.(t)}
                                    className={`w-full text-left flex items-center gap-3 p-2 rounded-xl hover:bg-white/10 transition
                    ${active ? 'bg-white/15 ring-1 ring-white/30' : ''}`}
                                >
                                    <img src={t.image} alt="" className="h-12 w-12 rounded-lg object-cover" />
                                    <div className="flex-1">
                                        <div className="font-medium leading-tight">{t.title}</div>
                                        <div className="text-sm text-zinc-400">{t.artist}</div>
                                    </div>
                                    <span className="text-zinc-400">❤</span>
                                </button>
                            </li>
                        )
                    })}
                </ul>
            </div>
        </section>
    )
}
