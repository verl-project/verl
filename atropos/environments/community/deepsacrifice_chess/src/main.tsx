import "./index.css";
import React from "react";
import { createRoot } from "react-dom/client";
import GameView from "./pages/GameView";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}

const root = createRoot(rootElement);
root.render(
  <React.StrictMode>
    <GameView />
  </React.StrictMode>,
);
