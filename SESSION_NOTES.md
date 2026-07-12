# Session Notes — resume point

> **To resume:** run `claude --continue` in this folder. If the session is gone,
> open Claude Code here and say: *"read SESSION_NOTES.md and continue the viva prep from Part 4."*

---

## What this project is
**Agentic News** — an AI agent that runs a news channel autonomously.
Repo: https://github.com/gaurmaitreyi-png/ai-post-generator
Article site: https://gaurmaitreyi-png.github.io/agentic-news/
Telegram bot: `@aipostgen_maitreyi_bot` · Channel: `Agentic News` (`agentic_newe_daily`)
Instagram: `@agentic_news_channel`

**Goal:** demo to professor (3rd-year minor project). I must be able to explain
every design choice and every line of code.

---

## Current state — ALL BUILT, TESTED, PUSHED

| Feature | Status |
|---|---|
| Telegram bot + channel posting (image, headline, "Read the full story" button) | Working |
| AI article generation (Gemini) + auto-published webpage on GitHub Pages | Working |
| Instagram posting (square image, curated caption, link-in-bio) | Working |
| **Fine-tuned DistilBERT clickbait classifier** (editorial quality gate) | Working |
| **Conversational agent** (LLM tool-calling — you just talk to it) | Working |
| **MCP server** (tools discovered over the protocol, not hard-coded) | Working |
| Facebook posting | REMOVED from UI (code dormant; needs `pages_manage_posts` on the Meta token) |

### The ML model (the key result)
- Dataset: **Chakraborty et al. 2016 (IEEE/ACM ASONAM)** — 32,000 real headlines,
  16k clickbait (BuzzFeed/Upworthy/ScoopWhoop), 16k genuine (NYT/Guardian/The Hindu).
  MUST be cited in the report.
- Fine-tuned `distilbert-base-uncased`, 3 epochs, lr 2e-5, batch 64, max_len 48.
- Split 80/10/10 stratified (25,600 / 3,200 / 3,200).
- **Test results: accuracy 99.03%, precision 98.76%, recall 99.31%, F1 99.03%**
- Confusion matrix: TN=1580, FP=20, FN=11, TP=1589
- Artifacts: `ml/metrics.json`, `ml/confusion_matrix.png`, `ml/training_curve.png`

---

## How to run
```bash
# start the bot (leave the window open — it dies when closed)
C:\Users\gaurm\miniconda3\python.exe bot.py

# retrain the model (only if needed; weights are gitignored)
C:\Users\gaurm\miniconda3\python.exe ml\train_clickbait.py
```
Python = the **miniconda base env** (NOT the `python` on PATH, which is a broken MS Store stub).

---

## Gotchas / things that bit us (good viva material)
1. **Instagram aspect ratio** — IG only accepts 4:5 to 1.91:1. News banners are too wide.
   Fix: always square-crop a Pexels image to 1080x1080.
2. **GitHub Pages build delay** — new article URLs 404 for ~1–2 min. Fix: poll until HTTP 200
   *before* posting, so the "Read more" link is never broken.
3. **NewsAPI category** — code sent `tech`, but the real slug is `technology` (returns an empty
   list, not an error). Fix: explicit category mapping.
4. **MCP server hung on every tool call** — loading the model lazily inside a request worker
   thread deadlocked under the server's event loop. Fix: preload on the main thread at startup.
   Also: inference moved to CPU (~20ms) — CUDA in a spawned subprocess was slow/fragile.
5. **Windows cp1252 crash** — emoji in `print()` killed startup. Fix: force UTF-8 stdout/stderr.
6. **Gemini free tier = 20 requests/day PER PROJECT PER MODEL** (not per key!). A new API key in
   the same project does NOT reset it. Currently on `gemini-3.1-flash-lite` (fresh quota).
   Model is configurable via `GEMINI_MODEL` in `.env`.

---

## Viva prep progress (we were mid-walkthrough)
- [x] Part 1 — big picture / 3-layer architecture
- [x] Part 2 — (skipped ahead)
- [x] Part 3 — code walkthrough: file map + `ml/train_clickbait.py` line by line
      (forward → loss → backward → clip → step → zero_grad; stratify; warmup; eval mode)
- [x] The maths: confusion matrix, accuracy/precision/recall/F1 derivations, training curves
- [x] Dataset provenance + BibTeX citation
- [ ] **Part 4 — `bot.py` (the application) walkthrough**   <-- RESUME HERE
- [ ] Part 5 — `agent.py`: how LLM tool-calling actually works
- [ ] Part 6 — MCP: what it is and why
- [ ] Part 7 — hard questions the professor will ask + answers

### Key framing to remember
Two separate, honest answers to "where's the AI/ML?":
1. **Generative AI (used, not built):** Gemini, via API — writes content, decides which tools to call.
2. **Machine Learning (built by me):** DistilBERT, fine-tuned on 32k headlines — 99.03% test accuracy.
Never blur the two.

---

## Outstanding / optional
- Update `PROJECT_REPORT.tex` — it still describes ML/agent/MCP as *planned*; they are now REAL.
  Add the real metrics + the dataset citation.
- Facebook: needs `pages_manage_posts` permission on the Meta token.
- The old leaked API keys (Telegram/NewsAPI/Pexels) are still in public git history — not rotated.
