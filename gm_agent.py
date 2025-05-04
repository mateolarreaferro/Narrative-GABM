# gm_agent.py
import time, uuid, math, functools
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import deque
from llm_utils import gen_oai                  # reuse your existing wrapper
from numpy.linalg import norm                 # tiny helper for cosine‑sim

# ───────────────────────────────────────────────────────────────────────────
#  Memory primitives
# ───────────────────────────────────────────────────────────────────────────
def cosine(a, b):
    return (a @ b) / (norm(a) * norm(b) + 1e-9)

@dataclass
class MemoryItem:
    """Lightweight episodic/semantic memory record."""
    text: str
    when: float               = field(default_factory=time.time)
    importance: float         = 1.0
    embedding: List[float]    = field(default_factory=list)

class MemoryBank:
    """
    Very small footprint: two deques keep recent episodes separate
    from longer‑term semantic facts.
    """
    def __init__(self, max_episodic=100, max_semantic=200):
        self.episodic = deque(maxlen=max_episodic)
        self.semantic = deque(maxlen=max_semantic)

    # — add —───────────────────────────────────────────────────────────────
    def add(self, text: str, kind="episodic", importance: float = 1.0, *, embed_fn):
        vec = embed_fn(text)
        item = MemoryItem(text=text, importance=importance, embedding=vec)
        (self.episodic if kind == "episodic" else self.semantic).appendleft(item)

    # — retrieve k best matches for a query vector —────────────────────────
    def retrieve(self, query: str, k=5, *, embed_fn) -> List[MemoryItem]:
        q_vec = embed_fn(query)
        scored = []
        for m in list(self.episodic) + list(self.semantic):
            score = cosine(q_vec, m.embedding) * m.importance
            scored.append((score, m))
        scored.sort(reverse=True)
        return [m for _, m in scored[:k]]

# ───────────────────────────────────────────────────────────────────────────
#  GMAgent
# ───────────────────────────────────────────────────────────────────────────
DIFFICULTY_PRESETS = {
    "easy":   dict(temp=0.7,   max_tokens=600, twist_rate=0.2),
    "normal": dict(temp=1.0,   max_tokens=800, twist_rate=0.4),
    "hard":   dict(temp=1.25,  max_tokens=900, twist_rate=0.6),
}

class GMAgent:
    def __init__(
        self,
        name: str,
        persona: str,
        difficulty: str = "normal",
        llm_model: str = "gpt-4o",
        embed_model: str = "text-embedding-3-small",
    ):
        self.name       = name
        self.persona    = persona.strip()
        self.difficulty = difficulty
        self.llm_model  = llm_model
        self.embed_model= embed_model
        self.mem        = MemoryBank()
        self.params     = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["normal"])

        # cheap caching of embeddings to avoid duplicate calls
        self._embed_cache: Dict[str, List[float]] = {}

    # — embedding helper —──────────────────────────────────────────────────
    def _embed(self, text: str) -> List[float]:
        if text in self._embed_cache:
            return self._embed_cache[text]
        res = gen_oai(
            [{"role": "user", "content": text}],
            model=self.embed_model,
            temperature=0,
        )  # returns 1536‑d list for this model
        vec = res if isinstance(res, list) else res["data"][0]["embedding"]
        self._embed_cache[text] = vec
        return vec

    # — public API: generate one GM turn —───────────────────────────────────
    def next_turn(
        self,
        scenario: Dict[str, Any],
        dialogue_history: List[str],
        player_instruction: str,
    ) -> str:
        """
        * scenario: one dict from scenarios.py
        * dialogue_history: list of all previous dialogue segments (strings)
        * player_instruction: the director’s latest order for their agent
        Returns raw multiline string in the same format your Room expects.
        """

        # ── 1. retrieve relevant memories ────────────────────────────────
        mem_snippets = self.mem.retrieve(
            player_instruction, k=4, embed_fn=self._embed
        )
        mem_block = "\n".join(f"- {m.text}" for m in mem_snippets) or "*(none)*"

        # ── 2. draft the LLM prompt ──────────────────────────────────────
        twist_directive = (
            f"There is a {int(self.params['twist_rate']*100)}% chance you inject "
            "a dramatic twist consistent with the scenario."
        )
        sys_prompt = (
            f"You are {self.name}, the Game Master with the persona:\n"
            f"\"{self.persona}\"\n\n"
            "Your job:\n"
            "• Narrate the world and control all NPCs.\n"
            "• Obey the required output template (exactly 1 GM line, "
            "player’s line, then 1‑3 NPC lines).\n"
            f"• Difficulty bias: **{self.difficulty.upper()}** "
            "(be harsher on players, more complex puzzles, etc.).\n"
            f"• {twist_directive}\n"
            "\nIf you declare the survival outcome, respect this rule:\n"
            f"{scenario['survival_rule']}\n"
        )

        hist = "\n".join(dialogue_history) or "*none yet*"
        user_prompt = (
            f"### Scenario\n{scenario['title']}\n\n"
            f"### Setup\n{scenario['setup']}\n\n"
            f"### Memory Bank\n{mem_block}\n\n"
            f"### Dialogue so far\n{hist}\n\n"
            f"### Director’s order to the player\n{player_instruction}\n\n"
            "### Produce the next turn now."
        )

        # ── 3. generate using your existing wrapper ──────────────────────
        raw = gen_oai(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            model=self.llm_model,
            temperature=self.params["temp"],
            max_tokens=self.params["max_tokens"],
        ).strip()

        # ── 4. learning: store interesting bits into memory ──────────────
        self._learn_from_turn(raw)

        return raw

    # — importance heuristic (very crude baseline) —────────────────────────
    def _score_importance(self, line: str) -> float:
        if any(k in line.lower() for k in ["survivor", "released", "stabilized"]):
            return 2.0
        return 1.0

    # — extract memory‑worthy chunks and save —─────────────────────────────
    def _learn_from_turn(self, raw_turn: str):
        for line in raw_turn.split("\n"):
            txt = line.strip()
            if not txt or txt.startswith("GM: …"):   # ignore typing indicator
                continue
            imp = self._score_importance(txt)
            self.mem.add(txt, importance=imp, embed_fn=self._embed)

# ───────────────────────────────────────────────────────────────────────────
#  Quick demo (delete or adapt in your tests)
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from scenarios import scenarios
    gm = GMAgent("Dr. Fate", "mischievous cosmic trickster", difficulty="hard")

    scen = scenarios[0]                             # lifeboat
    dialogue = []
    for _ in range(3):
        turn = gm.next_turn(scen, dialogue, "Focus on technical data about reactor failure.")
        dialogue.append(turn)
        print(turn, "\n" + "-" * 60 + "\n")
