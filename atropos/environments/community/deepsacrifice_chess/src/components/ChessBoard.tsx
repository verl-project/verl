import { Chess } from "chess.js";
import { Chessground } from "chessground";
import type React from "react";
import { useEffect, useRef } from "react";
import "./chessground.css";

interface ChessBoardProps {
  fen: string;
  onMove?: (from: string, to: string) => void;
  userTurn: boolean;
}

function getTurnColor(fen: string): "white" | "black" {
  const parts = fen.split(" ");
  return parts[1] === "b" ? "black" : "white";
}

function allSquares(): string[] {
  const files = "abcdefgh";
  const ranks = "12345678";
  const squares: string[] = [];
  for (let f = 0; f < 8; f++) {
    for (let r = 0; r < 8; r++) {
      squares.push(files[f] + ranks[r]);
    }
  }
  return squares;
}

function getDests(fen: string) {
  const chess = new Chess(fen);
  const dests: Record<string, string[]> = {};
  allSquares().forEach((sqr) => {
    const moves = chess.moves({ square: sqr as any, verbose: true });
    if (moves.length) dests[sqr] = moves.map((m: any) => m.to);
  });
  // Chessground expects a Map
  return new Map(Object.entries(dests));
}

const ChessBoard: React.FC<ChessBoardProps> = ({ fen, onMove, userTurn }) => {
  const boardRef = useRef<HTMLDivElement>(null);
  const groundRef = useRef<any>(null);
  const color = getTurnColor(fen);
  const dests = getDests(fen);

  useEffect(() => {
    if (boardRef.current) {
      if (!groundRef.current) {
        groundRef.current = Chessground(boardRef.current, {
          fen,
          orientation: "white",
          turnColor: color,
          movable: {
            color,
            dests: dests as any,
            free: false,
            events: {
              after: (orig: string, dest: string) => {
                if (onMove && userTurn) onMove(orig, dest);
              },
            },
          },
        });
      } else {
        groundRef.current.set({
          fen,
          turnColor: color,
          orientation: "white",
          movable: {
            color,
            dests: dests as any,
            free: false,
            events: {
              after: (orig: string, dest: string) => {
                if (onMove && userTurn) onMove(orig, dest);
              },
            },
          },
        });
      }
    }
  }, [
    fen,
    onMove,
    color,
    userTurn,
    JSON.stringify(Array.from(dests.entries())),
  ]);

  return (
    <div
      ref={boardRef}
      style={{ width: 400, height: 400 }}
    />
  );
};

export default ChessBoard;
