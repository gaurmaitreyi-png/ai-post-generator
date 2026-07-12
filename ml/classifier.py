"""
Inference wrapper around the fine-tuned DistilBERT clickbait classifier.

The bot uses this as an editorial quality gate: headlines predicted as clickbait
are blocked before anything is generated or published.

The model is loaded lazily (on first use) and kept in memory, so the bot starts
fast and only pays the load cost once. If the model has not been trained yet,
check_headline() fails open (treats the headline as genuine) so the bot keeps working.
"""

import contextlib
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

# Loading a HuggingFace model prints progress bars to stdout. When this runs inside the
# MCP server, stdout carries the JSON-RPC protocol, so any stray output corrupts it.
# Keep stdout pristine: silence the progress bars and send anything else to stderr.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clickbait_model")
LABELS = {0: "genuine", 1: "clickbait"}

_model = None
_tokenizer = None
_device = None
_lock = threading.Lock()


def _load():
    """Load the model once, on first use."""
    global _model, _tokenizer, _device
    if _model is not None:
        return True
    with _lock:
        if _model is not None:
            return True
        if not os.path.isdir(MODEL_DIR):
            logger.warning(f"Clickbait model not found at {MODEL_DIR}; run ml/train_clickbait.py")
            return False
        try:
            # Anything these libraries print must not land on stdout (see note above).
            with contextlib.redirect_stdout(sys.stderr):
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()

                # Inference runs on CPU by default: scoring one headline takes ~20ms, and
                # creating a CUDA context inside a spawned subprocess (e.g. the MCP server)
                # is slow and can hang. The GPU is only needed for training.
                use_cuda = os.getenv("CLICKBAIT_CUDA") == "1" and torch.cuda.is_available()
                _device = torch.device("cuda" if use_cuda else "cpu")
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
                _model.eval()
            logger.info(f"Clickbait classifier loaded on {_device}")
            return True
        except Exception as e:
            logger.error(f"Could not load clickbait classifier: {e}")
            return False


def check_headline(headline: str) -> dict:
    """Classify a headline.

    Returns {"label": "genuine"|"clickbait", "confidence": float, "is_clickbait": bool}.
    If the model is unavailable it fails open (label "genuine"), so publishing still works.
    """
    if not headline or not _load():
        return {"label": "genuine", "confidence": 0.0, "is_clickbait": False, "model_available": False}

    import torch
    with torch.no_grad():
        enc = _tokenizer(headline, truncation=True, padding="max_length",
                         max_length=48, return_tensors="pt").to(_device)
        probs = torch.softmax(_model(**enc).logits, dim=-1)[0]
    idx = int(probs.argmax())
    return {
        "label": LABELS[idx],
        "confidence": float(probs[idx]),
        "is_clickbait": idx == 1,
        "model_available": True,
    }


if __name__ == "__main__":
    # Quick manual check.
    for h in [
        "Which TV Female Friend Group Do You Belong In",
        "Bill Changing Credit Card Rules Is Sent to Obama With Gun Measure Included",
        "You Won't Believe What Happened Next",
        "ISRO successfully launches communication satellite into orbit",
    ]:
        r = check_headline(h)
        print(f"[{r['label']:>9}] {r['confidence']:.3f}  {h}")
