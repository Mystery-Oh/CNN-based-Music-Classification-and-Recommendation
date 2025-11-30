// src/components/Topbar.jsx
export default function Topbar({
   right,
   // 본문과 동일하게 쓸 컨테이너 폭/패딩
   containerClass = "max-w-7xl px-4 md:px-6 lg:px-10",
   spacerClass = "h-16",
   // 바(알약) 자체는 컨테이너 안에서 가득 차게
   barClass = "h-12 w-full",
                               }) {
    return (
        <header className={spacerClass}>
            <div className="fixed top-0 inset-x-0 z-30">
                <div className={`mx-auto ${containerClass} pt-4`}>
                    <div className={`glass ${barClass} px-4 flex items-center justify-between`}>
                        <div className="flex items-center gap-2">
                            <div className="h-7 w-7 rounded-xl bg-white/70" />
                            {/*<span className="font-semibold text-white/90">Muse UI</span>*/}
                        </div>
                        <div className="text-white/80">{right}</div>
                    </div>
                </div>
            </div>
        </header>
    )
}
