// // src/App.jsx
// import { BrowserRouter, Routes, Route } from "react-router-dom";
// import { useState } from "react"
// import SearchPage from "./pages/SearchPage.jsx"
// import PlayerPage from "./pages/PlayerPage.jsx"
//
// export default function App() {
//     const [current, setCurrent] = useState(null) // {id,title,artist,albumArt,url,...}
//     const [queue, setQueue] = useState([])       // 재생목록
//
//     const handlePlay = (track) => {
//         setCurrent(track)
//         // 큐에 없으면 추가
//         setQueue((q) => (q.find(t => t.id === track.id) ? q : [track, ...q]))
//     }
//
//     return (
//         <div className="min-h-screen">
//             {!current ? (
//                 <SearchPage onPlay={handlePlay} setQueue={setQueue} />
//             ) : (
//                 <PlayerPage
//                     current={current}
//                     queue={queue}
//                     onSelect={(t) => setCurrent(t)}
//                     onQueueChange={setQueue}
//                 />
//             )}
//         </div>
//     )
// }

// src/App.jsx (예시)
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useState } from "react";
import SearchPage from "./pages/SearchPage.jsx";
import PlayerPage from "./pages/PlayerPage.jsx";
import AnalyzePage from "./pages/AnalyzePage.jsx";

export default function App() {
    const [queue, setQueue] = useState([]);
    const [current, setCurrent] = useState(null);

    return (
        <BrowserRouter>
            <Routes>
                <Route
                    path="/"
                    element={
                        <SearchPage
                            setQueue={setQueue}
                            onPlay={setCurrent}
                        />
                    }
                />
                <Route
                    path="/player"
                    element={
                        <PlayerPage
                            current={current}
                            queue={queue}
                            onSelect={setCurrent}
                            onQueueChange={setQueue}
                        />
                    }
                />

                <Route
                    path="/analyze"
                    element={
                        <AnalyzePage />
                    }
                />

            </Routes>
        </BrowserRouter>
    );
}
