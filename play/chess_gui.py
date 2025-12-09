import pygame
import sys
import os
import chess
import threading
import time
from llm_player import load_llm, llm_pick_move

# ---------------- Config ----------------
AI_BASE = "Qwen/Qwen3-4B-Instruct-2507"  
AI_ADAPTER = "../out/qwen3-4b-qlora"           # path saved by train_sft_lora.py
PIECES_DIR = "assets"                         # folder containing piece images

# ---------------- Pygame Setup ----------------
pygame.init()
WIDTH, HEIGHT = 800, 400
SQUARE_SIZE = HEIGHT // 8
INFO_HEIGHT = 180

screen = pygame.display.set_mode((WIDTH, HEIGHT + INFO_HEIGHT))
pygame.display.set_caption("Chess GUI with LLM Player")

FONT = pygame.font.SysFont("Arial", 24)
SMALL_FONT = pygame.font.SysFont("Arial", 18)
BIG_FONT = pygame.font.SysFont("Arial", 40)

# ---------------- Colors ----------------
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
INFO_BG = (30, 30, 30)
TEXT_COLOR = (230, 230, 230)
RED = (200, 70, 70)
OVERLAY_BG = (12, 12, 12)

# ---------------- Load Piece Images ----------------
PIECE_KEYS = ["wP", "wN", "wB", "wR", "wQ", "wK",
              "bP", "bN", "bB", "bR", "bQ", "bK"]
pieces = {}
for key in PIECE_KEYS:
    path = os.path.join(PIECES_DIR, f"{key}.png")
    try:
        img = pygame.image.load(path).convert_alpha()
    except Exception as e:
        raise FileNotFoundError(f"Could not load piece image: {path}") from e
    pieces[key] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))

# ---------------- Board State ----------------
board = chess.Board()
selected_square = None
dragging_piece_key = None   # matches filename keys like wP, bN
dragging_rect = None
last_move = None

# Strategy/tactic text
current_strategy = ""
current_tactic = ""

# AI state
ai_thread = None
ai_result = None
ai_thinking = False
ai_attempt_log = []

# ---------------- Model Loading (Background) ----------------
tok = None
model = None
model_ready = False
model_error = None

def _load_model_worker():
    global tok, model, model_ready, model_error
    try:
        tok, model = load_llm(AI_BASE, AI_ADAPTER)  # 4-bit quant by default
        model_ready = True
        print("\n==============================")
        print(" Model successfully loaded")
        print(f" Base model: {AI_BASE}")
        print(f" Adapter:    {AI_ADAPTER}")
        print("==============================\n")
    except Exception as e:
        model_error = str(e)
        model_ready = False

model_thread = threading.Thread(target=_load_model_worker, daemon=True)
model_thread.start()

# ---------------- Drawing ----------------
def draw_board():
    # squares
    for r in range(8):
        for c in range(8):
            color = LIGHT if (r + c) % 2 == 0 else DARK
            pygame.draw.rect(screen, color, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    # highlight last move
    if last_move:
        from_sq = last_move.from_square
        to_sq = last_move.to_square
        for sq in (from_sq, to_sq):
            rr = 7 - (sq // 8)
            cc = sq % 8
            pygame.draw.rect(screen, (255, 255, 0), (cc * SQUARE_SIZE, rr * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

    # pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        rr = 7 - (square // 8)
        cc = square % 8
        color_key = "w" if piece.color == chess.WHITE else "b"
        key = color_key + piece.symbol().upper()
        img = pieces[key]
        screen.blit(img, (cc * SQUARE_SIZE, rr * SQUARE_SIZE))

    # dragging piece (follow mouse)
    if dragging_piece_key and dragging_rect:
        screen.blit(pieces[dragging_piece_key], dragging_rect)

def draw_info_panel():
    pygame.draw.rect(screen, INFO_BG, (0, HEIGHT, WIDTH, INFO_HEIGHT))
    y = HEIGHT + 10
    lines = [
        f"Strategy: {current_strategy or '—'}",
        f"Tactic: {current_tactic or '—'}",
        "",
        "AI Attempt Log (latest 5):"
    ]
    for line in lines:
        txt = SMALL_FONT.render(line, True, TEXT_COLOR)
        screen.blit(txt, (10, y))
        y += 22

    for entry in ai_attempt_log[-5:]:
        txt = SMALL_FONT.render(f"- {entry}", True, RED)
        screen.blit(txt, (20, y))
        y += 20

def draw_loading_overlay(start_time):
    screen.fill(OVERLAY_BG)
    title = BIG_FONT.render("Loading model…", True, TEXT_COLOR)
    sub = SMALL_FONT.render("This may take a moment, just once per run.", True, TEXT_COLOR)
    # animated dots
    elapsed = time.time() - start_time
    dots = "." * (int(elapsed * 2) % 4)
    animate = FONT.render(dots, True, TEXT_COLOR)

    title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20))
    sub_rect = sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20))
    anim_rect = animate.get_rect(center=(WIDTH // 2 + title_rect.width // 2 + 10, HEIGHT // 2 - 20))

    screen.blit(title, title_rect)
    screen.blit(sub, sub_rect)
    screen.blit(animate, anim_rect)

    if model_error:
        err = SMALL_FONT.render(f"Error: {model_error}", True, RED)
        err_rect = err.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
        screen.blit(err, err_rect)

# ---------------- Helpers ----------------
def mouse_to_square(pos):
    x, y = pos
    if y > HEIGHT:
        return None
    col = x // SQUARE_SIZE
    row = 7 - (y // 8 // SQUARE_SIZE)
    row = 7 - (y // SQUARE_SIZE)
    return chess.square(col, row)

def handle_player_move(from_sq, to_sq):
    global last_move, current_strategy, current_tactic
    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves:
        board.push(move)
        last_move = move
        current_strategy = ""
        current_tactic = ""
        return True
    return False

# ---------------- AI ----------------
def start_ai_move():
    """Kick off AI thinking in a background thread (requires model_ready)."""
    global ai_thread, ai_thinking, ai_result
    if not model_ready or model_error:
        return
    if board.is_game_over() or ai_thread is not None:
        return

    ai_thinking = True

    def _run_ai():
        global ai_thinking, ai_result
        try:
            move, strategy, tactic = llm_pick_move(
                board, tok, model,
                max_retries=6,
                first_without_legal=False 
            )
            ai_result = (move, strategy, tactic)
        except Exception as e:
            mv = next(iter(board.legal_moves))
            ai_result = (mv, "Fallback move due to error.", str(e))
        ai_thinking = False

    ai_thread = threading.Thread(target=_run_ai, daemon=True)
    print("\n=== AI MOVE REQUEST STARTED ===")
    print(f"FEN: {board.fen()}")
    print("Generating LLM move...\n")
    ai_thread.start()

# ---------------- Main Loop ----------------
running = True
clock = pygame.time.Clock()
loading_started = time.time()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Player input only after model loads (to avoid confusion during splash)
        if model_ready and not model_error and board.turn == chess.WHITE:
            if event.type == pygame.MOUSEBUTTONDOWN:
                square = mouse_to_square(event.pos)
                if square is not None and board.piece_at(square) and board.piece_at(square).color == chess.WHITE:
                    selected_square = square
                    piece = board.piece_at(square)
                    color_key = "w" if piece.color == chess.WHITE else "b"
                    dragging_piece_key = color_key + piece.symbol().upper()  # e.g., wP, bN
                    dragging_rect = pygame.Rect(event.pos[0]-30, event.pos[1]-30, SQUARE_SIZE, SQUARE_SIZE)

            if event.type == pygame.MOUSEMOTION and dragging_piece_key:
                dragging_rect.topleft = (event.pos[0]-30, event.pos[1]-30)

            if event.type == pygame.MOUSEBUTTONUP and dragging_piece_key:
                to_sq = mouse_to_square(event.pos)
                if selected_square is not None and to_sq is not None:
                    moved = handle_player_move(selected_square, to_sq)
                    if moved:
                        ai_thread = None
                        ai_result = None
                        start_ai_move()
                dragging_piece_key = None
                selected_square = None
                dragging_rect = None

    # Render
    if not model_ready and not model_error:
        draw_loading_overlay(loading_started)
    else:
        screen.fill((0, 0, 0))
        draw_board()
        draw_info_panel()

        # AI move section
        if model_ready and not model_error and board.turn == chess.BLACK and not board.is_game_over():
            if ai_thread is None and not ai_thinking:
                start_ai_move()
            if ai_result:
                mv, strat, tact = ai_result
                print("=== AI MOVE RECEIVED ===")
                print(f"Move:     {mv}")
                print(f"Strategy: {strat}")
                print(f"Tactic:   {tact}")
                print("========================\n")
                if mv in board.legal_moves:
                    board.push(mv)
                    last_move = mv
                    current_strategy = strat
                    current_tactic = tact
                ai_result = None
                ai_thread = None

        if model_error:
            err_txt = SMALL_FONT.render(f"Model load error: {model_error}", True, RED)
            screen.blit(err_txt, (10, HEIGHT + INFO_HEIGHT - 28))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
