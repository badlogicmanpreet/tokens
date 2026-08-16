"""Table-driven tests for the math verifier. Run: python3 -m pytest tests/"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.verify import grade

CASES = [
    ("14/3", r"\frac{14}{3}", True),
    ("0.5", "1/2", True),
    ("(14/3, 2/3)", "(14/3, 4/6)", True),
    ("15/3", "14/3", False),
    ("(0,5)", "[0,5]", False),          # open vs closed interval
    ("(0,5)", "(0,5)", True),
    (r"(\gcd(2,3), 5)", "(gcd(2,3), 5)", True),  # nested commas
    (r"\frac{\pi}{2}", "1.5707963267948966", True),
    (r"2\pi", "6.283185307179586", True),
    ("90^\\circ", "90", True),
    ("sqrt(2)", "1.4142135623730951", True),
    ("x+1", "1+x", True),
    ("1,234", "1234", True),
]

def test_grade_table():
    for pred, truth, want in CASES:
        assert grade(pred, truth) == want, (pred, truth, want)

def test_power_tower_does_not_hang():
    # must return (False) within the timeout, not wedge the process
    assert grade("9^{9^{9}}", "1") is False
