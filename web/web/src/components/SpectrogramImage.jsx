import { useRef, useState } from "react";

/** PNG 스펙트로그램 보기 (휠 줌, 드래그 팬) */
export default function SpectrogramImage({ src, className = "" }) {
    const wrapRef = useRef(null);
    const [scale, setScale] = useState(1);
    const [pos, setPos] = useState({ x: 0, y: 0 });
    const [drag, setDrag] = useState(null);

    const enc = (s) => (s ? encodeURI(s) : s);

    const onWheel = (e) => {
        e.preventDefault();
        const next = Math.max(0.5, Math.min(4, scale * (e.deltaY < 0 ? 1.1 : 0.9)));
        setScale(next);
    };

    const onDown = (e) => {
        const pt = e.touches ? e.touches[0] : e;
        setDrag({ sx: pt.clientX, sy: pt.clientY, ox: pos.x, oy: pos.y });
    };
    const onMove = (e) => {
        if (!drag) return;
        const pt = e.touches ? e.touches[0] : e;
        setPos({ x: drag.ox + (pt.clientX - drag.sx), y: drag.oy + (pt.clientY - drag.sy) });
    };
    const onUp = () => setDrag(null);

    return (
        <div
            ref={wrapRef}
            onWheel={onWheel}
            onMouseDown={onDown}
            onMouseMove={onMove}
            onMouseUp={onUp}
            onMouseLeave={onUp}
            onTouchStart={onDown}
            onTouchMove={onMove}
            onTouchEnd={onUp}
            className={`relative overflow-hidden rounded-xl bg-black/20 select-none ${className}`}
            style={{ cursor: drag ? "grabbing" : "grab" }}
        >
            <img
                src={src}
                alt="spectrogram"
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
                style={{
                    transform: `translate(-50%, -50%) scale(${scale}) translate(${pos.x / scale}px, ${pos.y / scale}px)`,
                    transformOrigin: "center center",
                    maxWidth: "100%",
                    pointerEvents: "none",
                }}
            />
        </div>
    );
}
