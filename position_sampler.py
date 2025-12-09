import argparse, random, json, os, glob, io, sys
import chess.pgn, chess
from tqdm import tqdm

import bz2, gzip

PHASE_PLY_THRESH = (10, 28)  # opening is the first 8 moves, middle game the next 17, endgame after that

def phase_from_ply(fullmove_number, ply_count):
    if ply_count < PHASE_PLY_THRESH[0]:
        return "opening"
    elif ply_count <= PHASE_PLY_THRESH[1]:
        return "middlegame"
    return "endgame"

def open_any(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    if path.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="ignore")
    if path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("zstandard not installed. pip install zstandard")
        dctx = zstd.ZstdDecompressor()
        fh = open(path, "rb")
        stream = dctx.stream_reader(fh)
        # TextIOWrapper will close underlying stream when closed
        return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    # plain text
    return open(path, "r", encoding="utf-8", errors="ignore")

def header_passes_filters(headers, min_elo, require_standard, min_clock_seconds):
    def parse_int(x):
        try:
            return int(x)
        except Exception:
            return None
    if min_elo is not None:
        w = parse_int(headers.get("WhiteElo"))
        b = parse_int(headers.get("BlackElo"))
        if w is None or b is None or w < min_elo or b < min_elo:
            return False
    if require_standard:
        variant = (headers.get("Variant") or headers.get("VariantType") or "Standard").strip().lower()
        if variant not in ("standard",):
            return False
    if min_clock_seconds is not None:
        # - "TimeControl": "600+0" (base+inc in seconds) or "-" (unknown)
        tc = headers.get("TimeControl")
        if tc and tc != "-":
            base = tc.split("+")[0]
            try:
                if int(base) < int(min_clock_seconds):
                    return False
            except Exception:
                pass  # if unparsable, don't filter on it
    return True

def iter_positions_from_game(game, step=2, max_per_game=8, rng=None):
    rng = rng or random
    board = game.board()
    headers = dict(game.headers)
    nodes = list(game.mainline())
    # Candidate ply indices along the game
    # Note that a ply is a half-move
    plies = list(range(0, len(nodes), step))
    # Randomly select up to max_per_game distinct plies
    rng.shuffle(plies)
    plies = sorted(plies[:max_per_game])
    ply_count = 0
    out = []
    for i, node in enumerate(nodes):
        move = node.move
        board.push(move)
        ply_count += 1
        if i not in plies:
            continue
        fen = board.fen()
        out.append({
            "fen": fen,
            "turn": "w" if board.turn == chess.WHITE else "b",
            "phase": phase_from_ply(board.fullmove_number, ply_count),
            "game_meta": {
                "Result": headers.get("Result"),
                "WhiteElo": headers.get("WhiteElo"), "BlackElo": headers.get("BlackElo"),
                "TimeControl": headers.get("TimeControl"),
            }
        })
    return out

def reservoir_sample_write(stream_iter, out_path, k=None, shard_size=None, seed=0):
    rng = random.Random(seed)
    if k is None:
        base = out_path
        count = 0
        shard_idx = 0
        w = None
        def open_writer(idx):
            path = base if idx == 0 else f"{base}.{idx}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return open(path, "w", encoding="utf-8")
        w = open_writer(shard_idx)
        for item in stream_iter:
            w.write(json.dumps(item) + "\n")
            count += 1
            if shard_size and (count % shard_size == 0):
                w.close()
                shard_idx += 1
                w = open_writer(shard_idx)
        if w:
            w.close()
        return count

    # Reservoir of size k
    reservoir = []
    n_seen = 0
    for item in stream_iter:
        n_seen += 1
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(1, n_seen)
            if j <= k:
                reservoir[j-1] = item

    # Write reservoir (optionally sharded)
    base = out_path
    os.makedirs(os.path.dirname(base), exist_ok=True)
    if shard_size:
        shard_idx = 0
        written = 0
        w = open(base, "w", encoding="utf-8")
        for i, item in enumerate(reservoir):
            if i > 0 and (i % shard_size == 0):
                w.close()
                shard_idx += 1
                w = open(f"{base}.{shard_idx}", "w", encoding="utf-8")
            w.write(json.dumps(item) + "\n")
            written += 1
        w.close()
        return written
    else:
        with open(base, "w", encoding="utf-8") as w:
            for item in reservoir:
                w.write(json.dumps(item) + "\n")
        return len(reservoir)

def stream_positions_from_paths(paths, step, max_per_game, game_stride, max_games,
                                min_elo, require_standard, min_clock_seconds, seed):
    rng = random.Random(seed)
    games_yielded = 0

    for p in paths:
        with open_any(p) as f:
            gi = 0
            with tqdm(desc=os.path.basename(p)) as pbar:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    gi += 1
                    pbar.update(1)
                    if game_stride and (gi - 1) % game_stride != 0:
                        continue
                    if not header_passes_filters(game.headers, min_elo, require_standard, min_clock_seconds):
                        continue
                    for rec in iter_positions_from_game(game, step=step, max_per_game=max_per_game, rng=rng):
                        yield rec
                    games_yielded += 1
                    if max_games and games_yielded >= max_games:
                        return

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pgn_file", help="Single PGN (supports .zst/.bz2/.gz)")
    src.add_argument("--pgn_glob", help="Glob for multiple PGNs")
    ap.add_argument("--out", default="data/normal/positions_raw.jsonl")
    ap.add_argument("--step", type=int, default=1, help="Sample every N plies within a game")
    ap.add_argument("--max_per_game", type=int, default=10, help="Max positions sampled per game")
    ap.add_argument("--game_stride", type=int, default=1, help="Take every Nth game (default 1 = every game)")
    ap.add_argument("--max_games", type=int, default=None, help="Stop after this many accepted games")
    ap.add_argument("--seed", type=int, default=0)

    # Filtering
    ap.add_argument("--min_elo", type=int, default=800, help="Require both players >= this Elo")
    ap.add_argument("--require_standard", action="store_true", help="Filter Variant to 'Standard'")
    ap.add_argument("--min_clock_seconds", type=int, default=30, help="Filter games with base time >= seconds")

    # Output control
    ap.add_argument("--reservoir", type=int, default=None, help="Keep exactly K positions via reservoir sampling")
    ap.add_argument("--shard_size", type=int, default=None, help="Write N lines per shard file")

    args = ap.parse_args()

    if args.pgn_glob:
        paths = sorted(glob.glob(args.pgn_glob))
    else:
        paths = [args.pgn_file]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    stream = stream_positions_from_paths(
        paths=paths,
        step=args.step,
        max_per_game=args.max_per_game,
        game_stride=args.game_stride,
        max_games=args.max_games,
        min_elo=args.min_elo,
        require_standard=args.require_standard,
        min_clock_seconds=args.min_clock_seconds,
        seed=args.seed
    )

    total = reservoir_sample_write(
        stream_iter=stream,
        out_path=args.out,
        k=args.reservoir,
        shard_size=args.shard_size,
        seed=args.seed
    )
    print(f"Wrote {total} positions to {args.out}" + (f" (+ shards)" if args.shard_size else ""))

if __name__ == "__main__":
    main()
