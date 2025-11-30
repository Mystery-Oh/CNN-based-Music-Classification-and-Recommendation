import "./index.css"
import React from "react"
import ReactDOM from "react-dom/client"
import { BrowserRouter, Routes, Route } from "react-router-dom"
import SearchPage from "./pages/SearchPage.jsx"
import PlayerPage from "./pages/PlayerPage.jsx"
import GlassWorkspace from "./pages/GlassWorkspace.jsx"
import AnalyzePage from "./pages/AnalyzePage.jsx";   // ← 만든 페이지

ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<SearchPage />} />
                <Route path="/player" element={<PlayerPage />} />
                <Route path="/analyze" element={<AnalyzePage />} />
                <Route path="/workspace" element={<GlassWorkspace />} />
                <Route path="*" element={<SearchPage />} />               {/* 404 fallback */}
            </Routes>
        </BrowserRouter>
    </React.StrictMode>
)
