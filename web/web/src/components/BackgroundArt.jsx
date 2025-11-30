export default function BackgroundArt({ src }) {
    const style = { backgroundImage: `url("${src || "/vite.svg"}")` }
    return (
        <>
            <div className="bg-album" style={style} />
            <div className="bg-overlay" />
        </>
    )
}
