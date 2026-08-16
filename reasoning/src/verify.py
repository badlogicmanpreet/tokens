"""Math answer verification pipeline.

Extracts, normalizes, and grades LLM-generated math answers against
reference solutions using symbolic comparison via SymPy.

Pipeline:  LLM output → extract boxed answer → normalize LaTeX → parse with SymPy → grade
"""

import re

from sympy import simplify
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError


# -----------------------------------------------------------------------
# 1. Answer extraction — pull the final answer from LLM output
# -----------------------------------------------------------------------

_RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def extract_boxed(text: str) -> str | None:
    """Find the content inside the last \\boxed{...} in *text*.

    Handles nested braces correctly so \\boxed{\\frac{14}{3}} returns
    '\\frac{14}{3}' rather than '\\frac{14'.
    """
    start = text.rfind(r"\boxed")
    if start == -1:
        return None

    i = start + len(r"\boxed")
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None

    i += 1
    depth = 1
    content_start = i

    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None
    return text[content_start : i - 1]


def extract_answer(text: str, fallback: str = "number_then_full") -> str:
    """Extract the final answer candidate from LLM output.

    Tries in order:
      1. Last \\boxed{...} expression
      2. Last number in the text (if fallback allows)
      3. The full text itself (if fallback='number_then_full')

    Args:
        text: raw LLM output
        fallback: 'number_then_full' | 'number_only' | 'none'
    """
    if not text:
        return ""

    boxed = extract_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")

    if fallback in ("number_then_full", "number_only"):
        nums = _RE_NUMBER.findall(text)
        if nums:
            return nums[-1]
        if fallback == "number_then_full":
            return text

    return ""


# -----------------------------------------------------------------------
# 2. Normalization — convert LaTeX to a canonical plain-text form
# -----------------------------------------------------------------------

_LATEX_SUBS = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
    (r"\\pi\b", "pi"),
    (r"π", "pi"),
    (r"\\infty\b", "oo"),
    (r"∞", "oo"),
    (r"\\times\b", "*"),
    (r"\\div\b", "/"),
    (r"\\(sin|cos|tan|log|ln|exp|gcd|lcm|min|max)\b", r"\1"),
]

_RE_SPECIAL_TOK = re.compile(r"<\|[^>]+?\|>")

_SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
}


def normalize(text: str) -> str:
    """Rewrite a math answer into a canonical plain-text form.

    Strips LaTeX markup, converts fractions/roots/exponents to
    calculator-style notation, and lowercases the result.

    Examples:
        \\dfrac{14}{3}      →  (14)/(3)
        \\sqrt{8}           →  sqrt(8)
        2^{10}              →  2**10
        1,234               →  1234
    """
    if not text:
        return ""

    # Strip special tokens like <|assistant|>
    text = _RE_SPECIAL_TOK.sub("", text).strip()

    # Strip leading multiple-choice labels: "c. 3" → "3"
    m = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if m:
        text = m.group(1)

    # Remove degree markers
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("°", "")

    # Unwrap \text{...}
    m = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if m:
        text = m.group("x")

    # Strip inline/display math wrappers
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # Apply LaTeX substitutions
    for pat, rep in _LATEX_SUBS:
        text = re.sub(pat, rep, text)

    # Convert unicode superscripts
    def _convert_sup(s, base=None):
        out = "".join(_SUPERSCRIPTS.get(c, c) for c in s)
        return f"{base}**{out}" if base else out

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: _convert_sup(m.group(2), base=m.group(1)),
        text,
    )
    text = _convert_sup(text)

    # Strip $, %, \%
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")

    # Convert \sqrt{...} → sqrt(...)
    text = re.sub(r"\\sqrt\s*\{([^}]*)\}", lambda m: f"sqrt({m.group(1)})", text)
    text = re.sub(r"\\sqrt\s+([^\\\s{}]+)", lambda m: f"sqrt({m.group(1)})", text)

    # Convert \frac{a}{b} → (a)/(b)
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda m: f"({m.group(1)})/({m.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda m: f"({m.group(1)})/({m.group(2)})",
        text,
    )

    # Convert ^ to **
    text = text.replace("^", "**")

    # Handle mixed numbers: "2 3/4" → "2+3/4"
    text = re.sub(r"(?<=\d)\s+(\d+/\d+)", lambda m: "+" + m.group(1), text)

    # Remove thousands separators: "1,234" → "1234"
    text = re.sub(r"(?<=\d),(?=\d\d\d(\D|$))", "", text)

    return text.replace("{", "").replace("}", "").strip().lower()


# -----------------------------------------------------------------------
# 3. Symbolic comparison via SymPy
# -----------------------------------------------------------------------

def _parse_sym(expr: str):
    """Parse a normalized math string into a SymPy expression."""
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,
                spp.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None


def _exprs_equal(a_str: str, b_str: str) -> bool:
    """Check if two normalized math strings are equivalent."""
    if a_str == b_str:
        return True

    a_sym, b_sym = _parse_sym(a_str), _parse_sym(b_str)
    if a_sym is not None and b_sym is not None:
        try:
            if simplify(a_sym - b_sym) == 0:
                return True
            # symbolic vs float (e.g. pi/2 vs 1.5707963...): compare numerically
            diff = (a_sym - b_sym).evalf()
            if diff.is_number and abs(float(diff)) < 1e-6:
                return True
        except (SympifyError, TypeError, ValueError, NotImplementedError,
                ZeroDivisionError, AttributeError, RecursionError,
                PolynomialError, OverflowError):
            pass
    return False


class _Timeout(Exception):
    pass


def _with_timeout(fn, seconds, default):
    """Run fn() with a hard time limit (SIGALRM; main thread, POSIX only).

    SymPy can effectively hang on adversarial input like 9**9**9 — a single
    such answer must not wedge a 500-problem eval or an RL training run.
    """
    import signal

    if not hasattr(signal, "SIGALRM"):
        return fn()  # non-POSIX: no guard available

    def _raise(signum, frame):
        raise _Timeout()

    prev = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        return fn()
    except _Timeout:
        return default
    except Exception:
        return default
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _split_top_level(inner: str) -> list[str]:
    """Split on commas at bracket depth 0 only, so '(gcd(2,3), 5)' -> 2 parts."""
    parts, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _split_tuple(text: str) -> list[str]:
    """Split a tuple/interval string '(a, b)' into its elements.

    The bracket style is preserved as a leading marker so that the open
    interval (0,5) never grades equal to the closed interval [0,5].
    """
    if not text:
        return []
    if (len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]):
        parts = _split_top_level(text[1:-1])
        if len(parts) >= 2 and all(parts):
            # bracket signature distinguishes (a,b) / [a,b] / (a,b] / [a,b)
            return [f"BRACKETS:{text[0]}{text[-1]}"] + parts
    return [text]


# -----------------------------------------------------------------------
# 4. Grading — the top-level function
# -----------------------------------------------------------------------

def grade(predicted: str, ground_truth: str) -> bool:
    """Grade a predicted answer against the ground truth.

    Handles scalar values, fractions, tuples, and various LaTeX formats.
    Returns True if the prediction is mathematically equivalent.

    Examples:
        grade("14/3", r"\\frac{14}{3}")        → True
        grade("(14/3, 2/3)", "(14/3, 4/6)")    → True
        grade("0.5", "1/2")                     → True
        grade("15/3", "14/3")                   → False
    """
    if predicted is None or ground_truth is None:
        return False

    gt_parts = _split_tuple(normalize(ground_truth))
    pred_parts = _split_tuple(normalize(predicted))

    if not gt_parts or not pred_parts:
        return False
    if len(gt_parts) != len(pred_parts):
        return False

    return _with_timeout(
        lambda: all(_exprs_equal(g, p) for g, p in zip(gt_parts, pred_parts)),
        seconds=5, default=False,
    )
