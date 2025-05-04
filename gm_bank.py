# gm_bank.py
from pathlib import Path
import json, pickle
from gm_agent import GMAgent

# ── 1. declarative templates ─────────────────────────────────────────────
GM_TEMPLATES = {
    "trickster": dict(
        name="Dr. Fate",
        persona="A mischievous cosmic trickster who delights in ironic twists.",
        difficulty="easy",
    ),
    "drill_sergeant": dict(
        name="Sgt. Steel",
        persona="Gruff military veteran; enforces rules to the letter.",
        difficulty="normal",
    ),
    "ai_overlord": dict(
        name="Mother Core",
        persona="Cold, hyper‑rational super‑AI that treats humans as variables.",
        difficulty="hard",
    ),
}

# ── 2. optional on‑disk persistence for each GM’s memory ────────────────
MEM_DIR = Path("gm_memories")
MEM_DIR.mkdir(exist_ok=True)

def _mem_path(code: str) -> Path:
    return MEM_DIR / f"{code}.pkl"

def _load_memory(code: str):
    path = _mem_path(code)
    return pickle.loads(path.read_bytes()) if path.exists() else None

def _save_memory(code: str, mem_bank):
    _mem_path(code).write_bytes(pickle.dumps(mem_bank))

# ── 3. public factory -----------------------------------------------------
def get_gm(code: str) -> GMAgent:
    """
    Returns a ready‑to‑use GMAgent instance.
    Memory is loaded from disk on first call and saved automatically
    whenever `next_turn()` is invoked.
    """
    if code not in GM_TEMPLATES:
        raise KeyError(f"GM '{code}' not defined in GM_TEMPLATES")

    cfg = GM_TEMPLATES[code]
    gm  = GMAgent(**cfg)                         # build the agent

    # plug in persisted memory (if any)
    stored = _load_memory(code)
    if stored:
        gm.mem = stored

    # monkey‑patch the agent so every learning step flushes to disk
    orig_learn = gm._learn_from_turn

    def learn_and_persist(raw_turn):
        orig_learn(raw_turn)
        _save_memory(code, gm.mem)

    gm._learn_from_turn = learn_and_persist
    return gm

# quick demo
if __name__ == "__main__":
    from scenarios import scenarios
    gm = get_gm("trickster")
    turn = gm.next_turn(scenarios[0], [], "Describe the reactor damage.")
    print(turn)
