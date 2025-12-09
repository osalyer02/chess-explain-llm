import argparse, os, json
import multiprocessing as mp
import chess, chess.engine
import pandas as pd
from tqdm import tqdm

# Convert engine score to centipawns with mate worth 32000
def score_to_cp(score: chess.engine.PovScore):
    if score.is_mate():
        m = score.mate()
        return (32000 if (m and m > 0) else -32000), True
    return score.score(), False

def analyze_fen(engine, fen, multipv=3, depth=16, movetime_ms=None):
    board = chess.Board(fen)
    legal_moves = [m.uci() for m in board.legal_moves]
    limit = chess.engine.Limit(depth=depth) if movetime_ms is None else chess.engine.Limit(time=movetime_ms/1000.0)
    info = engine.analyse(board, limit, multipv=multipv)
    info = sorted(info, key=lambda x: x.get("multipv", 1))
    best = info[0]
    def pv_to_uci(pv):
        return [m.uci() for m in pv] if pv else []
    best_move = best["pv"][0].uci()
    best_san = board.san(best["pv"][0])
    best_cp, best_is_mate = score_to_cp(best["score"].pov(board.turn))

    alt_moves, alt_scores_cp, pv_alt, deltas = [], [], [], []
    for cand in info[1:]:
        cp, _ = score_to_cp(cand["score"].pov(board.turn))
        alt_scores_cp.append(cp)
        alt_moves.append(cand["pv"][0].uci())
        pv_alt.append(pv_to_uci(cand["pv"]))
        deltas.append(best_cp - cp)

    return {
        "fen": fen,
        "turn": "w" if board.turn else "b",
        "phase": "unknown",
        "legal_moves": legal_moves,
        "multipv": len(info),
        "best_move": best_move,
        "best_san": best_san,
        "sf_depth": best.get("depth", 0),
        "best_score_cp": best_cp,
        "best_is_mate": best_is_mate,
        "pv_best": [m.uci() for m in best["pv"]],
        "alt_moves": alt_moves,
        "alt_scores_cp": alt_scores_cp,
        "pv_alt": pv_alt,
        "score_deltas_cp": deltas
    }

def worker(task):
    fen, eng_path, args = task
    try:
        with chess.engine.SimpleEngine.popen_uci(eng_path) as engine:
            engine.configure({"Threads": args.threads, "Hash": args.hash})
            return analyze_fen(engine, fen, multipv=args.multipv, depth=args.depth, movetime_ms=args.movetime_ms)
    except Exception as e:
        return {"fen": fen, "error": str(e)}

def append_to_parquet(df, out_path):
    # Check if file exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(out_path):
        df.to_parquet(out_path, index=False)
    else:
        # Append via pyarrow
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(out_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input file of FENs from position_sampler.py")
    ap.add_argument("--engine_path", default="stockfish")
    ap.add_argument("--out_parquet", default="data/output/normal/labeled_raw.parquet")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--movetime_ms", type=int, default=None)
    ap.add_argument("--multipv", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count()//2))
    ap.add_argument("--flush_interval", type=int, default=4000, help="Write partial results every N evaluations")
    args = ap.parse_args()

    # Read all FENs
    fens = []
    with open(args.in_jsonl, "r") as f:
        for line in f:
            obj = json.loads(line)
            fens.append(obj["fen"])
    total_fens = len(fens)

    tasks = [(fen, args.engine_path, args) for fen in fens]
    pool = mp.Pool(args.procs)

    batch = []
    processed = 0
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)

    print(f"Starting analysis on {total_fens:,} positions using {args.procs} processes")
    with tqdm(total=total_fens, desc="Labeling") as pbar:
        for out in pool.imap_unordered(worker, tasks):
            batch.append(out)
            processed += 1
            pbar.update(1)
            if len(batch) >= args.flush_interval:
                df = pd.DataFrame(batch)
                append_to_parquet(df, args.out_parquet)
                batch.clear()
                pbar.set_postfix_str(f"flushed {processed:,}")

    # Write remaining
    if batch:
        df = pd.DataFrame(batch)
        append_to_parquet(df, args.out_parquet)

    pool.close(); pool.join()
    print(f"Wrote {processed:,} positions to {args.out_parquet}")

if __name__ == "__main__":
    main()
