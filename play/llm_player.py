# llm_player.py
import json
from typing import Optional, Tuple, List
import chess
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------- Loaders ----------------
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

def load_llm(
    base_model: str,
    adapter_dir: str,
    device_map: str = "auto",
    torch_dtype = torch.bfloat16,
    quantized: bool = True,          # <- enable 4-bit by default
    compute_dtype = torch.bfloat16, 
):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # safer for generation with batching

    if quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",        
            bnb_4bit_use_double_quant=True,  
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    # Attach LoRA adapter on top of the (quantized) base
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        is_trainable=False,  # inference
    )
    model.eval()
    return tok, model


# ---------------- Templates (mirror training) ----------------
SYS_TRAIN = (
    'You are a chess assistant. Given a FEN, output JSON with keys "move" '
    '(a UCI like e2e4 or e7e8q), "strategy" (one sentence), and "tactic" '
    '(one sentence). Do not include any extra text.'
)

def _format_legal_list_ucis(legal_moves: List[str]) -> str:
    # EXACT format used in training: "<LEGAL>UCI: a5a1, a5a2, ...</LEGAL>"
    return "UCI: " + ", ".join(legal_moves)

def _build_prompt_with_legal(fen: str, legal_moves: List[str], feedback: str | None = None) -> str:
    legal_str = _format_legal_list_ucis(legal_moves)
    fb = f"\n<FEEDBACK>{feedback}</FEEDBACK>" if feedback else ""
    return (
        f"<SYS>{SYS_TRAIN}</SYS>\n"
        f"<POS>FEN: {fen}</POS>\n"
        f"<LEGAL>{legal_str}</LEGAL>\n"
        "<TASK>Pick the best move and explain.</TASK>"
        f"{fb}"
    )

def _build_prompt_initial_blind(fen: str) -> str:
    return (
        f"<SYS>{SYS_TRAIN}</SYS>\n"
        f"<POS>FEN: {fen}</POS>\n"
        "<TASK>Pick the best move and explain.</TASK>"
    )

def _repair_json(txt: str) -> Optional[dict]:
    start = txt.find("{"); end = txt.find("}")
    if start >= 0 and end > start:
        try:
            return json.loads(txt[start:end+1])
        except Exception:
            return None
    return None

# ---------------- Inference ----------------
def llm_pick_move(
    board: chess.Board, tok, model,
    max_new_tokens=128, temperature=0.2, top_p=0.9,
    max_retries=6, first_without_legal=False 
) -> Tuple[chess.Move, str, str]:
    legal_list = [m.uci() for m in board.legal_moves]
    if not legal_list:
        raise ValueError("No legal moves")

    attempts = 0
    bad_reasons: List[str] = []
    used_legal_prompt_once = not first_without_legal

    while attempts < max_retries:
        attempts += 1
        if first_without_legal and attempts == 1:
            prompt = _build_prompt_initial_blind(board.fen())
        else:
            used_legal_prompt_once = True
            feedback = bad_reasons[-1] if bad_reasons else None
            prompt = _build_prompt_with_legal(board.fen(), legal_list, feedback)

        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids, do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, eos_token_id=getattr(tok, "eos_token_id", None)
        )
        text = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

        obj = _repair_json(text)

        if not obj or "move" not in obj:
            bad_reasons.append("Malformed or missing JSON/move")
            print(" JSON parse failed — retrying.")
            continue

        mv_text = obj["move"].strip()
        # Prefer UCI (training used UCI), but accept SAN if needed
        try:
            if len(mv_text) >= 4 and mv_text[0] in "abcdefgh":
                move = chess.Move.from_uci(mv_text)
            else:
                move = board.parse_san(mv_text)
        except Exception:
            bad_reasons.append(f"Unparseable move: {mv_text}")
            continue

        if move not in board.legal_moves:
            bad_reasons.append(f"Illegal move: {mv_text}")
            print(f" Illegal move produced: {mv_text} — retrying.\n")
            continue

        return move, obj.get("strategy", ""), obj.get("tactic", "")

    # Fallback so the GUI never stalls
    fallback = next(iter(board.legal_moves))
    reason = ("Model repeatedly returned invalid output."
              + (" (LEGAL list was provided)" if used_legal_prompt_once else ""))
    return fallback, "Fallback: chose a safe legal move.", reason
