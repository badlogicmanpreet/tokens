"""Download pre-trained model weights and tokenizer files."""

from pathlib import Path

from tqdm import tqdm
import urllib.request

_HF_BASE = "https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main"
_MIRROR  = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"

_ASSETS = {
    "base": {
        "tokenizer": "tokenizer-base.json",
        "weights":   "qwen3-0.6B-base.pth",
    },
    "reasoning": {
        "tokenizer": "tokenizer-reasoning.json",
        "weights":   "qwen3-0.6B-reasoning.pth",
    },
}


def fetch_file(url: str, dest: Path, fallback_url: str | None = None):
    """Stream-download *url* into *dest* with a progress bar.

    Skips the download when a local file of matching size already exists.
    Falls back to *fallback_url* on failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as resp:
                remote_size = int(resp.headers.get("Content-Length", 0))
            if remote_size and dest.stat().st_size == remote_size:
                print(f"\u2713 {dest} already up-to-date")
                return
        except Exception:
            pass

    try:
        _stream(url, dest)
    except Exception as exc:
        if fallback_url:
            print(f"Primary URL failed ({exc}), trying mirror...")
            _stream(fallback_url, dest)
        else:
            raise


def _stream(url: str, dest: Path):
    with urllib.request.urlopen(urllib.request.Request(url)) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(8192):
                f.write(chunk)
                bar.update(len(chunk))


def download_weights(variant: str = "base", out_dir: str | Path = "qwen3",
                     tokenizer_only: bool = False):
    """Download Qwen3 0.6B assets.

    Args:
        variant: 'base' or 'reasoning'
        out_dir: local directory to save into
        tokenizer_only: skip the large weight file if True
    """
    out = Path(out_dir)
    assets = _ASSETS[variant]

    fetch_file(
        f"{_HF_BASE}/{assets['tokenizer']}",
        out / assets["tokenizer"],
        f"{_MIRROR}/{assets['tokenizer']}",
    )
    if not tokenizer_only:
        fetch_file(
            f"{_HF_BASE}/{assets['weights']}",
            out / assets["weights"],
            f"{_MIRROR}/{assets['weights']}",
        )
