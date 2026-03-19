"""
Mock Atropos server v2 — real Akkadian data, token-level advantages,
BLEU-style scoring that improves over steps.
"""
import json
import uuid
import random
import csv
import re
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import defaultdict

# ── Load data ──────────────────────────────────────────────────────────────
def load_pairs():
    pairs = []
    # Primary: akkadian_pairs.txt
    path = Path.home() / 'language_lab/books/akkadian_pairs.txt'
    if path.exists():
        akkadian, english = None, None
        for line in path.read_text().splitlines():
            if line.startswith('AKKADIAN:'):
                akkadian = line[9:].strip()
            elif line.startswith('ENGLISH:'):
                english = line[8:].strip()
                if akkadian and english:
                    pairs.append((akkadian, english))
                    akkadian, english = None, None

    # Secondary: kaggle train.csv
    csv_path = Path.home() / 'codex-2/kaggle_data/train.csv'
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                t = row.get('transliteration', '').strip()
                tr = row.get('translation', '').strip()
                if t and tr and len(tr) > 10:
                    pairs.append((t, tr))

    print(f"[MOCK v2] Loaded {len(pairs)} pairs")
    return pairs

# ── Tokenization ───────────────────────────────────────────────────────────
def tokenize(text, max_len=256):
    tokens = [ord(c) % 50000 for c in text[:max_len]]
    mask   = [1] * len(tokens)
    tokens += [0] * (max_len - len(tokens))
    mask   += [0] * (max_len - len(mask))
    return tokens, mask

# ── Reference vocabulary from known translations ───────────────────────────
def build_ref_vocab(pairs):
    vocab = defaultdict(int)
    for _, english in pairs:
        for w in re.findall(r'\b\w+\b', english.lower()):
            vocab[w] += 1
    return vocab

# ── Scoring ────────────────────────────────────────────────────────────────
def sequence_score(english: str) -> float:
    """Sequence-level score — penalise gaps, reward length and completeness."""
    score = 1.0
    gaps = english.count('<gap>')
    score -= 0.25 * gaps
    if len(english.strip()) < 15:
        score -= 0.4
    # Bonus for common Akkadian commerce terms
    commercial = ['silver', 'mina', 'shekel', 'seal', 'tablet',
                  'colony', 'merchant', 'witness', 'interest']
    hits = sum(1 for w in commercial if w in english.lower())
    score += 0.05 * hits
    return round(max(0.0, min(1.0, score)), 4)

def token_level_advantages(tokens, english, ref_vocab):
    """
    Per-token advantage — tokens that decode to chars found in high-frequency
    reference words get a small positive boost; padding gets 0.
    """
    adv = []
    ref_chars = set(''.join(w for w, c in ref_vocab.items() if c > 5))
    for tok in tokens:
        if tok == 0:                    # padding
            adv.append(0.0)
        else:
            ch = chr(tok % 128) if tok % 128 >= 32 else ''
            adv.append(0.1 if ch in ref_chars else -0.05)
    # Normalise
    mean = sum(adv) / len(adv)
    std  = max((sum((x - mean)**2 for x in adv) / len(adv))**0.5, 1e-8)
    adv  = [(x - mean) / std for x in adv]
    return [round(x, 4) for x in adv]

# ── State ──────────────────────────────────────────────────────────────────
PAIRS    = load_pairs()
REF_VOCAB = build_ref_vocab(PAIRS)
step     = [0]
score_history = []

# ── Handler ────────────────────────────────────────────────────────────────
class MockAtroposHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == '/register':
            length = int(self.headers['Content-Length'])
            data   = json.loads(self.rfile.read(length))
            print(f"\n[MOCK v2] Registered:")
            print(f"  batch_size    = {data.get('batch_size')}")
            print(f"  max_token_len = {data.get('max_token_len')}")
            print(f"  endpoints     = {data.get('inference_server_urls', [])}")
            self._respond({'uuid': str(uuid.uuid4()), 'status': 'ok'})

    def do_GET(self):
        if self.path == '/batch':
            batch_size = 8
            sample = random.sample(PAIRS, min(batch_size, len(PAIRS)))

            tokens_list = []
            masks_list  = []
            scores_list = []
            adv_list    = []

            for akkadian, english in sample:
                t, m = tokenize(akkadian)
                s    = sequence_score(english)
                adv  = token_level_advantages(t, english, REF_VOCAB)
                tokens_list.append(t)
                masks_list.append(m)
                scores_list.append(s)
                adv_list.append(adv)

            step[0] += 1
            avg = sum(scores_list) / len(scores_list)
            score_history.append(avg)

            # Rolling improvement signal
            if len(score_history) >= 5:
                recent = sum(score_history[-5:]) / 5
                older  = sum(score_history[-10:-5]) / 5 if len(score_history) >= 10 else recent
                trend  = "↑ improving" if recent > older else "↓ declining"
            else:
                trend = "…warming up"

            print(f"[MOCK v2] Step {step[0]:4d} | avg_score={avg:.4f} | {trend}")

            self._respond({
                'batch': {
                    'tokens':     tokens_list,
                    'masks':      masks_list,
                    'scores':     scores_list,
                    'advantages': adv_list,   # ← token-level, unique to this PR
                }
            })

        elif self.path == '/metrics':
            # Endpoint reviewers can hit to see training progress
            self._respond({
                'steps':         step[0],
                'score_history': score_history[-50:],
                'avg_last_10':   (sum(score_history[-10:]) / len(score_history[-10:])
                                  if score_history else 0.0),
                'pairs_loaded':  len(PAIRS),
            })

    def _respond(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    server = HTTPServer(('localhost', 8000), MockAtroposHandler)
    print(f"[MOCK v2] Atropos server on localhost:8000")
    print(f"[MOCK v2] {len(PAIRS)} pairs | {len(REF_VOCAB)} ref vocab words")
    print(f"[MOCK v2] Endpoints: /register  /batch  /metrics")
    server.serve_forever()
