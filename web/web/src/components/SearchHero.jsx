import { useState } from "react"

export default function SearchHero({ onSelectTrack, onSearch }) {
    const [q, setQ] = useState("")

    const submit = (e) => {
        e.preventDefault()
        onSearch?.(q)
    }

    return (
        <div className="flex items-center justify-center min-h-screen">
            <div className="w-full max-w-3xl px-6">
                <div className="text-center mb-8">
                    <h1 className="text-4xl md:text-5xl font-semibold tracking-tight mb-6">
                        지금 무슨 생각을 하시나요?
                    </h1>
                    <p className="text-zinc-600 dark:text-zinc-300">원하는 곡이나 아티스트를 검색해 보세요.</p>
                </div>

                <form onSubmit={submit} className="flex items-center gap-3 bg-white/10 dark:bg-zinc-900/40 backdrop-blur-xl rounded-2xl border border-white/20 dark:border-zinc-800 p-3 shadow-md">
                    <span className="text-xl">＋</span>
                    <input
                        className="flex-1 bg-transparent outline-none placeholder:text-zinc-400 text-lg"
                        placeholder="무엇이든 물어보세요"
                        value={q}
                        onChange={e=>setQ(e.target.value)}
                    />
                    <button className="px-4 py-2 rounded-xl bg-white/70 dark:bg-zinc-800 hover:bg-white/90 dark:hover:bg-zinc-700 transition">
                        검색
                    </button>
                </form>

                {/* (옵션) 간단한 추천 아이템 */}
                <div className="mt-6 flex flex-wrap gap-2">
                    {['Taylor Swift', 'NewJeans', 'Coldplay'].map((t)=>(
                        <button key={t} onClick={()=>onSearch?.(t)} className="px-3 py-1 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 text-sm">
                            {t}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    )
}
