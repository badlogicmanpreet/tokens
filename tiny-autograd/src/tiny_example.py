"""The chapter's worked example, runnable: e, f, b with Loss = ((e + f) * b)^2.

Forward:  a = e + f = 5.0    c = a * b = 20.0    Loss = c^2 = 400.0
Backward: Loss.grad = 1
          c.grad = 2 * c * Loss.grad          = 40
          a.grad = b * c.grad                 = 160
          b.grad = a * c.grad                 = 200
          e.grad = f.grad = a.grad            = 160

With lr = 0.001, one hundred simultaneous SGD updates take the Loss from
400 to roughly 1.25 — run it and watch.
"""
from datastructure import Value


def main():
    e, f, b = Value(2.0), Value(3.0), Value(4.0)

    for step in range(100):
        # forward pass (rebuild the graph each step)
        a = e + f
        c = a * b
        loss = c ** 2

        # backward pass
        e.grad = f.grad = b.grad = 0.0
        loss.backward()

        if step == 0:
            print(f"step 0: loss = {loss.data:.1f}  "
                  f"e.grad = {e.grad:.1f}  f.grad = {f.grad:.1f}  b.grad = {b.grad:.1f}")

        # SGD update
        for p in (e, f, b):
            p.data -= 0.001 * p.grad

    a = e + f
    c = a * b
    loss = c ** 2
    print(f"step 100: loss = {loss.data:.4f}  "
          f"(e = {e.data:.3f}, f = {f.data:.3f}, b = {b.data:.3f})")


if __name__ == "__main__":
    main()
