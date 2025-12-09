import argparse, pandas as pd, chess
from collections import Counter
import numpy as np

CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}

def compute_legal_moves_for_fen(fen: str):
    try:
        b = chess.Board(fen)
        return sorted([m.uci() for m in b.legal_moves])
    except Exception:
        return []

def features_from_board(board: chess.Board):
    """Extract simple strategic features for auto_explanations."""
    pawns = board.pieces(chess.PAWN, board.turn)
    mobility = sum(1 for _ in board.legal_moves)
    center_ctrl = sum(1 for m in board.legal_moves if m.to_square in CENTER)

    pawn_files = [chess.square_file(sq) for sq in pawns]
    doubled = any(pawn_files.count(f) > 1 for f in range(8))

    # ---- Passed pawn check (manual) ----
    def is_passed_pawn(sq):
        """Return True if pawn at sq is passed."""
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        direction = 1 if board.color_at(sq) == chess.WHITE else -1
        opp_color = not board.color_at(sq)
        for df in (-1, 0, 1):
            f = file + df
            if not (0 <= f < 8):
                continue
            r = rank + direction
            while 0 <= r < 8:
                sq2 = chess.square(f, r)
                piece = board.piece_at(sq2)
                if piece and piece.piece_type == chess.PAWN and piece.color == opp_color:
                    return False
                r += direction
        return True

    passed = any(is_passed_pawn(sq) for sq in pawns)

    def pawn_shield_intact(color):
        ksq = board.king(color)
        if ksq is None:
            return False
        rank = chess.square_rank(ksq)
        file = chess.square_file(ksq)
        front_rank = rank + (1 if color == chess.WHITE else -1)
        shield = 0
        for df in (-1, 0, 1):
            f = file + df
            if 0 <= f < 8 and 0 <= front_rank < 8:
                sq = chess.square(f, front_rank)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    shield += 1
        return shield >= 2

    open_files = []
    for f in range(8):
        file_squares = [chess.square(f, r) for r in range(8)]
        white_pawn_on_file = any(
            board.piece_type_at(sq) == chess.PAWN and board.color_at(sq) == chess.WHITE
            for sq in file_squares
        )
        black_pawn_on_file = any(
            board.piece_type_at(sq) == chess.PAWN and board.color_at(sq) == chess.BLACK
            for sq in file_squares
        )
        if not white_pawn_on_file and not black_pawn_on_file:
            open_files.append(f)

    return {
        "mobility": mobility,
        "center_ctrl": center_ctrl,
        "doubled_pawns": doubled,
        "passed_pawn": passed,
        "open_files": open_files,
        "our_shield_ok": pawn_shield_intact(board.turn),
    }


import numpy as np  # make sure this is at the top of the file


def render_strategy_for_move(board: chess.Board, pv_uci):
    """
    Render strategic motifs, but only if they are made *more true* by the top engine move.

    If pv_uci is empty / invalid, falls back to board-only strategic description.
    """
    # ---- Normalise pv_uci safely ----
    if pv_uci is None:
        feat = features_from_board(board)
        return render_strategy_from_features(feat)

    # If it's a numpy array, convert to list
    if isinstance(pv_uci, np.ndarray):
        pv_uci = pv_uci.tolist()

    # If it's not already a list/tuple, try to coerce it to a list
    if not isinstance(pv_uci, (list, tuple)):
        try:
            pv_uci = list(pv_uci)
        except Exception:
            feat = features_from_board(board)
            return render_strategy_from_features(feat)

    # Safe emptiness check
    if len(pv_uci) == 0:
        feat = features_from_board(board)
        return render_strategy_from_features(feat)

    # Take the first move
    first = pv_uci[0]
    if not isinstance(first, str) or len(first) < 4:
        feat = features_from_board(board)
        return render_strategy_from_features(feat)

    try:
        move = chess.Move.from_uci(first)
    except Exception:
        feat = features_from_board(board)
        return render_strategy_from_features(feat)

    if move not in board.legal_moves:
        feat = features_from_board(board)
        return render_strategy_from_features(feat)

    # ---- Compute features before and after the move ----
    before = features_from_board(board)
    b_after = board.copy()
    b_after.push(move)
    after = features_from_board(b_after)

    parts = []

    # --- increase central control: central control strictly increases ---
    if after["center_ctrl"] > before["center_ctrl"]:
        parts.append("increases central control")

    # --- use open files for rooks: number of open files increases ---
    if len(after["open_files"]) > len(before["open_files"]):
        parts.append("uses open files for rooks")

    # --- advance the passed pawn: the move actually advances a passed pawn ---
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        file = chess.square_file(move.from_square)
        rank = chess.square_rank(move.from_square)
        direction = 1 if piece.color == chess.WHITE else -1
        opp_color = not piece.color

        def is_passed_from_here():
            for df in (-1, 0, 1):
                f = file + df
                if not (0 <= f < 8):
                    continue
                r = rank + direction
                while 0 <= r < 8:
                    sq2 = chess.square(f, r)
                    p = board.piece_at(sq2)
                    if p and p.piece_type == chess.PAWN and p.color == opp_color:
                        return False
                    r += direction
            return True

        if is_passed_from_here():
            to_rank = chess.square_rank(move.to_square)
            if (piece.color == chess.WHITE and to_rank > rank) or (
                piece.color == chess.BLACK and to_rank < rank
            ):
                parts.append("advances the passed pawn")

    # --- avoid further pawn weaknesses: don't create new doubled pawns ---
    if after["doubled_pawns"] and not before["doubled_pawns"]:
        # Move created a new weakness -> don't claim it's avoiding pawn weaknesses
        pass
    else:
        if before["doubled_pawns"]:
            parts.append("avoids further pawn weaknesses")

    # --- improve king safety: pawn shield becomes OK after the move ---
    if not before["our_shield_ok"] and after["our_shield_ok"]:
        parts.append("improves king safety")

    # --- leverage superior piece activity: mobility goes up and is high ---
    if after["mobility"] >= 30 and after["mobility"] > before["mobility"]:
        parts.append("leverages superior piece activity")

    if not parts:
        return "None."

    return "This move " + ", ".join(parts) + "."


def render_strategy_from_features(feat):
    """Original feature-only rendering: used as a fallback when we don't have a clear best move."""
    parts = []
    if feat["center_ctrl"] >= 2:
        parts.append("increases central control")
    if feat["open_files"]:
        parts.append("uses open files for rooks")
    if feat["passed_pawn"]:
        parts.append("advances the passed pawn")
    if feat["doubled_pawns"]:
        parts.append("avoids further pawn weaknesses")
    if not feat["our_shield_ok"]:
        parts.append("improves king safety")
    if feat["mobility"] >= 30:
        parts.append("leverages superior piece activity")
    if not parts:
        return "None."
    return "This move " + ", ".join(parts) + "."


import numpy as np

def pv_tactics_tags(board: chess.Board, pv_uci):
    tags = []

    # Null check
    if pv_uci is None:
        return tags
    
    if isinstance(pv_uci, np.ndarray):
        pv_uci = pv_uci.tolist()

    if not isinstance(pv_uci, (list, tuple)):
        try:
            pv_uci = list(pv_uci)
        except Exception:
            return tags
    
    if len(pv_uci) == 0:
        return tags

    u = pv_uci[0]
    if not isinstance(u, str) or len(u) < 4:
        return tags

    try:
        m = chess.Move.from_uci(u)
    except Exception:
        return tags

    b = board.copy()
    if m not in b.legal_moves:
        return tags

    # Capture / check motifs from the first move
    if b.is_capture(m):
        tags.append("capture")
    if b.gives_check(m):
        tags.append("check")

    # Fork pattern (knight) – immediate fork after the first move
    piece = b.piece_at(m.from_square)
    b.push(m)
    if piece and piece.piece_type == chess.KNIGHT:
        attacks = list(b.attacks(m.to_square))
        valuable = sum(
            1 for sq in attacks
            if (p := b.piece_at(sq)) and p.piece_type in (chess.ROOK, chess.QUEEN)
        )
        if valuable >= 2:
            tags.append("fork")

    return sorted(set(tags))


def render_tactics(tags, mate_flag, score_cp):
    parts = []
    if mate_flag: parts.append("forces mate")
    if "check" in tags: parts.append("gives check")
    if "capture" in tags: parts.append("wins material")
    if "fork" in tags: parts.append("creates a fork")
    if not parts:
        return "None."
    return "This move " + ", ".join(parts) + "."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--recompute_legal", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    strat = []
    tact = []
    tags = []
    need_legal = args.recompute_legal or "legal_moves" not in df.columns
    legal_moves_col = []
    for fen, pv, mate, sc in zip(df["fen"], df["pv_best"], df["best_is_mate"], df["best_score_cp"]):
        board = chess.Board(fen)
        ttags = pv_tactics_tags(board, pv)
        strat.append(render_strategy_for_move(board, pv))
        tact.append(render_tactics(ttags, mate, sc))
        tags.append(ttags)
        if need_legal:
            legal_moves_col.append(compute_legal_moves_for_fen(fen))
    df["strat_text"] = strat
    df["tact_text"] = tact
    df["tags"] = tags
    if need_legal:
        df["legal_moves"] = legal_moves_col
    df.to_parquet(args.out_parquet, index=False)
    print("Wrote", args.out_parquet, "rows:", len(df))

if __name__ == "__main__":
    main()
