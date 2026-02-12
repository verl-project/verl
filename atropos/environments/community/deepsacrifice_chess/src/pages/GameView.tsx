import type React from "react";
import { useEffect, useState } from "react";
import ChessBoard from "../components/ChessBoard";

interface MoveData {
  move: { from: string; to: string; san?: string };
  fen: string;
  reward: number;
  llmFeedback: { score: string; justification: string };
}

const initialFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const GameView: React.FC = () => {
  const [fen, setFen] = useState<string>(initialFen);
  const [moveHistory, setMoveHistory] = useState<MoveData[]>([]);
  const [feedback, setFeedback] = useState<null | {
    score: string;
    justification: string;
    reward: number;
  }>(null);
  const [userTurn, setUserTurn] = useState<boolean>(true);
  const [gameOver, setGameOver] = useState<boolean>(false);
  const [scoring, setScoring] = useState(false);
  const [scored, setScored] = useState(false);

  // Start a new game (reset backend state and local state)
  const startNewGame = async () => {
    const res = await fetch("/api/reset", { method: "POST" });
    const data = await res.json();
    setFen(data.fen || initialFen);
    setMoveHistory([]);
    setFeedback(null);
    setUserTurn(true);
    setGameOver(false);
    setScored(false);
  };

  useEffect(() => {
    startNewGame();
  }, []);

  // Helper to call backend for a move
  const makeMove = async (from: string, to: string) => {
    const res = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from, to }),
    });
    const data = await res.json();

    // If backend returns moves array (user + agent)
    if (Array.isArray(data.moves)) {
      setMoveHistory((prev) => [
        ...prev,
        ...data.moves.map((m: any) => ({
          move: m.move,
          fen: m.fen,
          reward: m.reward,
          llmFeedback: m.llmFeedback,
        })),
      ]);
      // Set FEN to the last move's FEN
      const lastMove = data.moves[data.moves.length - 1];
      setFen(lastMove.fen);
      setFeedback({
        score: lastMove.llmFeedback?.score,
        justification: lastMove.llmFeedback?.justification,
        reward: lastMove.reward,
      });
      if (data.done) setGameOver(true);
      return data;
    }

    // fallback for old response shape
    setFen(data.fen);
    setFeedback({
      score: data.llmFeedback?.score,
      justification: data.llmFeedback?.justification,
      reward: data.reward,
    });
    setMoveHistory((prev) => [
      ...prev,
      {
        move: { from, to },
        fen: data.fen,
        reward: data.reward,
        llmFeedback: data.llmFeedback,
      },
    ]);
    if (data.done) setGameOver(true);
    return data;
  };

  // User move handler
  const handleMove = async (from: string, to: string) => {
    if (!userTurn || gameOver) return;
    setUserTurn(false);
    const data = await makeMove(from, to);
    // After both moves (user + agent), re-enable user input if game not over
    if (!data.done) {
      setTimeout(() => {
        setUserTurn(true);
      }, 500); // Small delay for UX
    }
  };

  // Score the game after it ends
  const scoreGame = async () => {
    setScoring(true);
    try {
      const res = await fetch("/api/game/llm_feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ moves: moveHistory }),
      });
      const data = await res.json();
      if (Array.isArray(data.moves)) {
        setMoveHistory(data.moves);
        setScored(true);
      }
    } finally {
      setScoring(false);
    }
  };

  return (
    <div
      style={{
        background: "#fff",
        minHeight: "100vh",
        height: "100vh",
        fontFamily: "monospace",
        color: "#111",
        padding: 0,
        margin: 0,
        border: "8px solid #111",
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Toolbar */}
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          borderBottom: "4px solid #111",
          background: "#fff",
          padding: "0 12px",
          minHeight: 40,
          height: 40,
          gap: 16,
        }}
      >
        <span
          style={{
            fontWeight: 900,
            fontSize: 18,
            letterSpacing: 1,
            marginRight: 16,
            textTransform: "uppercase",
            userSelect: "none",
          }}
        >
          <span style={{ fontWeight: 400 }}>Deep</span>Sacrifice
        </span>
        <button
          onClick={startNewGame}
          style={{
            fontWeight: 900,
            fontSize: 12,
            border: "2px solid #111",
            background: "#fff",
            color: "#111",
            padding: "2px 10px",
            cursor: "pointer",
            textTransform: "uppercase",
            boxShadow: "none",
            borderRadius: 0,
            height: 28,
            lineHeight: "24px",
            alignSelf: "center",
          }}
        >
          New Game
        </button>
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          flex: 1,
          minHeight: 0,
          minWidth: 0,
          justifyContent: "space-between",
          alignItems: "stretch",
          padding: 16,
          gap: 24,
        }}
      >
        <div
          style={{
            flex: "0 0 auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <ChessBoard
            fen={fen}
            onMove={handleMove}
            userTurn={userTurn && !gameOver}
          />
        </div>
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
          }}
        >
          <div
            style={{
              marginBottom: 8,
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              gap: 12,
            }}
          >
            {!scored ? (
              <button
                onClick={scoreGame}
                disabled={scoring}
                style={{
                  fontWeight: 700,
                  fontSize: 14,
                  border: "2px solid #111",
                  background: scoring ? "#eee" : "#fff",
                  color: scoring ? "#aaa" : "#111",
                  padding: "2px 10px",
                  cursor: scoring ? "not-allowed" : "pointer",
                  textTransform: "uppercase",
                  borderRadius: 0,
                  height: 28,
                  lineHeight: "24px",
                  alignSelf: "center",
                  opacity: scoring ? 0.7 : 1,
                }}
              >
                {scoring ? "Scoring..." : "Score Game"}
              </button>
            ) : (
              <button
                onClick={startNewGame}
                style={{
                  fontWeight: 700,
                  fontSize: 14,
                  border: "2px solid #111",
                  background: "#fff",
                  color: "#111",
                  padding: "2px 10px",
                  cursor: "pointer",
                  textTransform: "uppercase",
                  borderRadius: 0,
                  height: 28,
                  lineHeight: "24px",
                  alignSelf: "center",
                }}
              >
                New Game
              </button>
            )}
          </div>
          {gameOver && (
            <div
              style={{
                color: "#fff",
                background: "#111",
                marginBottom: 8,
                padding: 8,
                fontWeight: 900,
                border: "2px solid #111",
                borderRadius: 0,
                textAlign: "center",
                letterSpacing: 1,
              }}
            >
              Game Over
            </div>
          )}
          {feedback && null}
          <div
            style={{
              flex: 1,
              minHeight: 0,
              display: "flex",
              flexDirection: "column",
              fontSize: 13,
              lineHeight: 1.3,
            }}
          >
            <div
              style={{
                flex: 1,
                minHeight: 0,
                overflow: "hidden",
                display: "flex",
                flexDirection: "column",
              }}
            >
              <div
                style={{
                  flex: 1,
                  minHeight: 0,
                  overflowY: "auto",
                  width: "100%",
                }}
              >
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    background: "#fff",
                    fontFamily: "monospace",
                    tableLayout: "fixed",
                    fontSize: 12,
                  }}
                >
                  <thead>
                    <tr>
                      <th
                        style={{
                          border: "2px solid #111",
                          padding: 6,
                          fontWeight: 900,
                          background: "#eee",
                          textAlign: "left",
                          width: "8%",
                          minWidth: 40,
                          maxWidth: 60,
                        }}
                      >
                        #
                      </th>
                      <th
                        style={{
                          border: "2px solid #111",
                          padding: 6,
                          fontWeight: 900,
                          background: "#eee",
                          textAlign: "left",
                          width: "32%",
                          minWidth: 80,
                          maxWidth: 120,
                        }}
                      >
                        Move
                      </th>
                      <th
                        style={{
                          border: "2px solid #111",
                          padding: 6,
                          fontWeight: 900,
                          background: "#eee",
                          textAlign: "left",
                          width: "30%",
                          minWidth: 60,
                          maxWidth: 100,
                        }}
                      >
                        Reward
                      </th>
                      <th
                        style={{
                          border: "2px solid #111",
                          padding: 6,
                          fontWeight: 900,
                          background: "#eee",
                          textAlign: "left",
                          width: "30%",
                          minWidth: 60,
                          maxWidth: 100,
                        }}
                      >
                        LLM Score
                      </th>
                      <th
                        style={{
                          border: "2px solid #111",
                          padding: 6,
                          fontWeight: 900,
                          background: "#eee",
                          textAlign: "left",
                          width: "50%",
                          minWidth: 100,
                          maxWidth: 300,
                        }}
                      >
                        Justification
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {moveHistory.map((m, i) => (
                      <tr key={i}>
                        <td
                          style={{
                            border: "2px solid #111",
                            padding: 6,
                            width: "8%",
                            minWidth: 40,
                            maxWidth: 60,
                          }}
                        >
                          {i + 1}
                        </td>
                        <td
                          style={{
                            border: "2px solid #111",
                            padding: 6,
                            width: "32%",
                            minWidth: 80,
                            maxWidth: 120,
                          }}
                        >
                          {m.move.from}-{m.move.to}
                        </td>
                        <td
                          style={{
                            border: "2px solid #111",
                            padding: 6,
                            width: "30%",
                            minWidth: 60,
                            maxWidth: 100,
                          }}
                        >
                          {m.reward === null || m.reward === undefined
                            ? "–"
                            : m.reward}
                        </td>
                        <td
                          style={{
                            border: "2px solid #111",
                            padding: 6,
                            width: "30%",
                            minWidth: 60,
                            maxWidth: 100,
                          }}
                        >
                          {m.llmFeedback?.score === null ||
                          m.llmFeedback?.score === undefined
                            ? "–"
                            : m.llmFeedback.score}
                        </td>
                        <td
                          style={{
                            border: "2px solid #111",
                            padding: 6,
                            width: "50%",
                            minWidth: 100,
                            maxWidth: 300,
                          }}
                        >
                          {m.llmFeedback?.justification === null ||
                          m.llmFeedback?.justification === undefined
                            ? "–"
                            : m.llmFeedback.justification}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameView;
