# --- keep imports, add chess & numpy use if not already present ---
import argparse
import json
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd
import chess  # NEW

TEMPLATE_SYS = (
    "You are a chess assistant. Given a FEN, output JSON with keys "
    '"move" (a UCI like e2e4 or e7e8q), "strategy" (one sentence), and "tactic" (one sentence). '
    "Do not include any extra text."
)

def _norm_list(x):
    """Normalize a column that may be list/tuple/np.array/scalar -> list or None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return list(x)
    except Exception:
        return None

def _legal_from_fen(fen: str) -> list[str]:
    try:
        b = chess.Board(fen)
        return sorted(m.uci() for m in b.legal_moves)
    except Exception:
        return []

def row_to_prompt(fen: str, legal_moves: list[str] | None) -> str:
    lm = legal_moves or []
    # stable ordering for determinism; CSV on one line for compactness
    lm_csv = ", ".join(sorted(lm))
    return (
        f"<SYS>{TEMPLATE_SYS}</SYS>\n"
        f"<POS>FEN: {fen}</POS>\n"
        f"<LEGAL>UCI: {lm_csv}</LEGAL>\n"
        f"<TASK>Pick the best move and explain.</TASK>"
    )

def row_to_response(move: str, strategy: str, tactic: str) -> str:
    payload = {
        "move": move,
        "strategy": strategy or "",
        "tactic": tactic or "",
    }
    return json.dumps(payload, ensure_ascii=False)

def iter_examples(
    df: pd.DataFrame,
    min_margin_cp: int,
    max_margin_cp: int | None,
) -> Iterable[dict]:
    """
    Yield {"prompt":..., "response":...} dicts from a labeled+annotated DataFrame.
    Applies margin filtering using `score_deltas_cp` if present.
    """
    has_deltas = "score_deltas_cp" in df.columns
    has_legal = "legal_moves" in df.columns

    for _, r in df.iterrows():
        fen = r.get("fen")
        move = r.get("best_move")
        strat = r.get("strat_text", "")
        tact = r.get("tact_text", "")

        if not fen or not move:
            continue

        # Margin filter
        if has_deltas:
            deltas = _norm_list(r.get("score_deltas_cp"))
            max_delta = max(deltas) if deltas else None
            if max_delta is not None:
                if max_delta < min_margin_cp:
                    continue
                if max_margin_cp is not None and max_delta > max_margin_cp:
                    continue

        # Legal moves for this row
        legals = None
        if has_legal:
            legals = _norm_list(r.get("legal_moves"))
        if not legals:
            legals = _legal_from_fen(fen)

        yield {
            "prompt": row_to_prompt(fen, legals),
            "response": row_to_response(move, strat, tact),
        }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True, help="data/positions_labeled.parquet")
    ap.add_argument("--out_jsonl", default="data/sft_train.jsonl")
    ap.add_argument("--out_val_jsonl", default=None, help="optional validation file")
    ap.add_argument("--min_margin_cp", type=int, default=50, help="keep samples with max(best - alt) >= this")
    ap.add_argument("--max_margin_cp", type=int, default=None, help="optionally cap margin to avoid only-easy data")
    ap.add_argument("--dedupe_by_fen", action="store_true", help="drop duplicate FENs before exporting")
    ap.add_argument("--sample", type=int, default=None, help="optional cap on number of training rows")
    ap.add_argument("--val_fraction", type=float, default=0.05, help="0..0.5; split off this fraction as validation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    df = pd.read_parquet(args.in_parquet)

    # Basic cleanup
    if "error" in df.columns:
        df = df[df["error"].isna()]
    # Ensure required columns exist
    for c in ("fen", "best_move"):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if args.dedupe_by_fen and "fen" in df.columns:
        df = df.drop_duplicates(subset=["fen"], keep="first")

    # Shuffle once for splitting/sampling
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Build examples (generator -> list)
    examples = list(iter_examples(df, args.min_margin_cp, args.max_margin_cp))

    # Optional sampling
    if args.sample is not None and args.sample < len(examples):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(examples), size=args.sample, replace=False)
        examples = [examples[i] for i in sorted(idx)]

    # Optional validation split
    if args.val_fraction and 0.0 < args.val_fraction < 0.5:
        n_total = len(examples)
        n_val = int(n_total * args.val_fraction)
        val = examples[:n_val]
        train = examples[n_val:]
    else:
        train, val = examples, None

    # Write train
    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        for ex in train:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write val if requested
    if args.out_val_jsonl and val is not None:
        with open(args.out_val_jsonl, "w", encoding="utf-8") as w:
            for ex in val:
                w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} examples to {args.out_jsonl}")
    if args.out_val_jsonl and val is not None:
        print(f"Wrote {len(val)} examples to {args.out_val_jsonl}")


if __name__ == "__main__":
    main()
