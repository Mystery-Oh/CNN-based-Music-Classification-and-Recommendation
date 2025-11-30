export default function NeonBackground() {
    return (
        <div className="fixed inset-0 -z-10 bg-[#0a0b13] overflow-hidden vignette">
            {/* 큰 보라+핑크 블롭 (좌하단) */}
            <div
                className="absolute -left-[12vw] bottom-[-8vh] w-[50vw] h-[50vw] rounded-full
                   blur-3xl opacity-80 animate-float"
                style={{
                    background:
                        "radial-gradient(circle at 30% 30%, #ff66cc 0%, #8b5cf6 40%, transparent 70%)",
                    filter: "blur(60px)",
                }}
            />

            {/* 시안+보라 블롭 (우상단) */}
            <div
                className="absolute right-[-10vw] -top-[12vh] w-[55vw] h-[55vw] rounded-full
                   blur-3xl opacity-80 animate-float-slow"
                style={{
                    background:
                        "radial-gradient(circle at 60% 40%, #22d3ee 0%, #60a5fa 25%, #a78bfa 45%, transparent 70%)",
                    filter: "blur(70px)",
                }}
            />

            {/* 작은 시안 포인트 (카드 뒤 하이라이트 느낌) */}
            <div
                className="absolute left-[48%] top-[28%] w-[22vw] h-[22vw] rounded-full
                   blur-2xl opacity-80 animate-float"
                style={{
                    background:
                        "radial-gradient(circle at 50% 50%, #22d3ee 0%, #06b6d4 40%, transparent 70%)",
                    filter: "blur(50px)",
                }}
            />

            {/* 어두운 반투명 원 (우측) */}
            <div
                className="absolute right-[12vw] top-[38%] w-[18vw] h-[18vw] rounded-full
                   bg-white/5 mix-blend-multiply blur-xl"
                style={{ backdropFilter: "blur(2px)" }}
            />

            {/* 아주 작은 보라 점 (중앙 하단) */}
            <div
                className="absolute left-[52%] top-[62%] w-[8vw] h-[8vw] rounded-full
                   blur-2xl opacity-80"
                style={{
                    background:
                        "radial-gradient(circle at 50% 50%, #a78bfa 0%, transparent 60%)",
                    filter: "blur(40px)",
                }}
            />
        </div>
    )
}
