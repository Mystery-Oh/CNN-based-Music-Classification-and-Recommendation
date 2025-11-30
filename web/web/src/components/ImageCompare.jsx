// 상/하 1:1로 나눠 한 화면에 정확히 들어오도록 조정
export default function ImageCompare({ leftSrc, rightSrc, className = "" }) {
    return (
        <div className={`relative ${className} min-h-0`}>
            <div
                className="
          relative h-full p-3 md:p-4 grid gap-4 min-h-0
          grid-rows-[minmax(0,1fr)_minmax(0,1fr)]
        "
            >
                {/* Left */}
                <div className="relative min-h-0 rounded-2xl overflow-hidden bg-black/20 border border-white/10 flex items-center justify-center">
                    {leftSrc ? (
                        <img src={leftSrc} alt="left" className="w-full h-auto max-h-full" />
                    ) : (
                        <Empty label="Left" />
                    )}
                </div>

                {/* Right */}
                <div className="relative min-h-0 rounded-2xl overflow-hidden bg-black/20 border border-white/10 flex items-center justify-center">
                    {rightSrc ? (
                        <img src={rightSrc} alt="right" className="w-full h-auto max-h-full" />
                    ) : (
                        <Empty label="Right" />
                    )}
                </div>
            </div>
        </div>
    );
}

function Empty({ label }) {
    return (
        <div className="text-white/60 text-sm select-none">
      <span className="inline-block bg-white/10 px-2 py-1 rounded-md mr-2">
        {label}
      </span>
            No image
        </div>
    );
}
