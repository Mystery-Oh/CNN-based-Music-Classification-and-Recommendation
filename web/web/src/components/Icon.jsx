export default function Icon({
                                 name,
                                 size = 20,             // px
                                 stroke = 1.8,          // 선형 아이콘 굵기
                                 className = "",
                                 title,                 // 접근성(툴팁/스크린리더용)
                                 ...props
                             }) {
    const common = { fill: "none", stroke: "currentColor", strokeWidth: stroke, strokeLinecap: "round", strokeLinejoin: "round" };

    const map = {
        // ▶
        play: (
            <path d="M8 5v14l11-7-11-7z" fill="currentColor" />
        ),
        // ⏸
        pause: (
            <g fill="currentColor">
                <rect x="6" y="5" width="4" height="14" rx="1" />
                <rect x="14" y="5" width="4" height="14" rx="1" />
            </g>
        ),
        // ⏭
        next: (
            <g fill="currentColor">
                <path d="M6 7l8 5-8 5V7z" />
                <rect x="16" y="6" width="2" height="12" rx="1" />
            </g>
        ),
        // ⏮
        prev: (
            <g fill="currentColor">
                <path d="M18 7l-8 5 8 5V7z" />
                <rect x="6" y="6" width="2" height="12" rx="1" />
            </g>
        ),
        // ••• (세로)
        dotsVertical: (
            <g fill="currentColor">
                <circle cx="12" cy="5" r="2" />
                <circle cx="12" cy="12" r="2" />
                <circle cx="12" cy="19" r="2" />
            </g>
        ),
        // Heart(아웃라인)
        heart: (
            <path
                {...common}
                d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-.99-1.06a5.5 5.5 0 1 0-7.78 7.78L12 21.35l8.77-8.96a5.5 5.5 0 0 0 .07-7.78z"
            />
        ),
        // Heart(채움)
        heartFill: (
            <path
                d="M12 21s-6.72-4.35-9.33-7.47C.94 10.6 2.2 7.5 4.9 6.6c1.6-.55 3.42-.1 4.6 1.02L12 10l2.5-2.38c1.18-1.12 3-1.57 4.6-1.02 2.7.9 3.96 4 2.23 6.93C18.72 16.65 12 21 12 21z"
                fill="currentColor"
            />
        ),
        // 🔍
        search: (
            <>
                <circle cx="11" cy="11" r="7" {...common} />
                <path d="M20 20l-3.2-3.2" {...common} />
            </>
        ),
        // ←
        chevronLeft: <path d="M15 18l-6-6 6-6" {...common} />,
    };

    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            width={size}
            height={size}
            className={className}
            role={title ? "img" : "presentation"}
            aria-hidden={title ? undefined : true}
            {...props}
        >
            {title ? <title>{title}</title> : null}
            {map[name]}
        </svg>
    );
}
