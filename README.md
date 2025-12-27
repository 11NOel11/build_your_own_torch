# build_your_own_torch 🔥

**A ground-up, educational reimplementation of core PyTorch concepts** — autograd, tensors, neural network modules, and optimizers — built step by step from first principles.

**build_your_own_torch** is a learning-focused project whose goal is **understanding, not performance**.

Instead of treating deep learning frameworks as black boxes, this repo reconstructs their core ideas from scratch:

- **how automatic differentiation actually works**
- **how computation graphs are built and traversed**
- **how gradients flow via the chain rule**
- **how tensors, layers, and optimizers fit together**

If you've ever wondered what really happens when you call `loss.backward()` or `optimizer.step()`, this project is for you.

---

## ✨ Philosophy

Most deep learning libraries optimize for:
- speed
- hardware acceleration
- massive scale

**This project optimizes for:**
- **clarity**
- **minimalism**
- **correctness**
- **learning**

Everything is intentionally:
- **small**
- **explicit**
- **readable**
- **hackable**

The code is written to be **read, modified, and experimented with**.

---

## 🚀 Features (Current Status)

### ✅ Stage 1 — Scalar Autograd (Implemented)

- A `Value` class representing a scalar node in a computation graph
- Automatic construction of computation graphs through operator overloading
- Reverse-mode automatic differentiation (backpropagation)
- DFS-based topological sorting of computation graphs
- Local gradient rules via per-operation `_backward()` closures

**Supported operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Unary ops: negation
- Nonlinearities: `tanh`, `exp`

**End‑to‑end training example:**
- Single tanh neuron
- Mean squared error loss
- Gradient descent loop
- Verified convergence

*This stage alone reproduces the core logic of PyTorch's autograd engine, but for scalars.*

---

## 🧠 Example: Scalar Autograd

```python
from build_your_own_torch.autograd_scalar import Value

x = Value(2.0)
y = Value(3.0)

f = x * y + x     # builds a computation graph
f.backward()      # runs backpropagation

print(f)          # Value(data=8.0, grad=1.0)
print(x.grad)     # 4.0
print(y.grad)     # 2.0
```

**Behind the scenes:**
- a computation graph is built automatically
- `.backward()` performs a reverse topological traversal
- each node applies its local chain-rule contribution

---

## 🧪 Example: Training a Tiny Neuron

```python
from build_your_own_torch.autograd_scalar import Value

x1, x2 = Value(1.0), Value(-2.0)
w1, w2 = Value(0.5), Value(-1.0)
b = Value(0.0)

learning_rate = 0.1

for step in range(50):
    # forward pass
    n = (x1 * w1 + x2 * w2 + b).tanh()
    loss = (n - Value(0.0)) ** 2

    # zero gradients
    for p in (w1, w2, b):
        p.grad = 0.0

    # backward pass
    loss.backward()

    # gradient descent update
    for p in (w1, w2, b):
        p.data -= learning_rate * p.grad

    print(step, loss.data)
```

*This uses your own autograd engine, not PyTorch.*

---

## 🗺️ Roadmap

This project is built incrementally, with each stage layering new abstractions on top of the previous one.

### Stage 1 — Scalar Autograd ✅
- ✅ `Value` class
- ✅ Computation graph construction
- ✅ Backpropagation with DFS + topological sort
- ✅ Scalar ops and nonlinearities
- ✅ Tiny neuron training example

### Stage 2 — Tensor Autograd (NumPy Backend)
- ⬜ `Tensor` class (`data: np.ndarray`, `grad: np.ndarray`)
- ⬜ Elementwise operations with broadcasting
- ⬜ Matrix multiplication
- ⬜ Reduction ops (sum, mean)
- ⬜ Reverse-mode autograd for tensors

### Stage 3 — Neural Network Modules
- ⬜ `Module` base class
- ⬜ Parameter registration
- ⬜ Layers: `Linear`, `ReLU`, `Tanh`
- ⬜ Loss functions (MSE, Cross‑Entropy)

### Stage 4 — Optimizers
- ⬜ `Optimizer` base class
- ⬜ SGD
- ⬜ Momentum
- ⬜ RMSProp
- ⬜ Adam
- ⬜ AdamW
- ⬜ (Stretch) modern optimizers like Muon or Lion

### Stage 5 — Experiments & Examples
- ⬜ Linear regression
- ⬜ Multi-layer perceptron
- ⬜ Toy classification datasets
- ⬜ Optimizer comparisons

---

## 📦 Installation (Using uv)

This project uses [**uv**](https://github.com/astral-sh/uv) for fast, reproducible Python environments.

### 1. Clone the repository

```bash
git clone https://github.com/11NOel11/build_your_own_torch.git
cd build_your_own_torch
```

### 2. Install uv (if needed)

```bash
pip install uv
```

### 3. Sync the environment

```bash
uv sync
```

This will:
- create an isolated virtual environment
- install dependencies defined in `pyproject.toml`

### 4. Run code inside the environment

```bash
uv run python examples/01_scalar_autograd_demo.py
```

---

## 📁 Project Structure

```
build_your_own_torch/
│
├── build_your_own_torch/
│   ├── __init__.py
│   ├── autograd_scalar.py     # Stage 1: scalar autograd
│   ├── tensor.py              # Stage 2: tensor autograd (planned)
│   ├── nn/                    # Stage 3: neural network modules
│   └── optim/                 # Stage 4: optimizers
│
├── examples/
│   ├── 01_scalar_autograd_demo.py  # scalar neuron training demo
│   └── ...
│
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

---

## 📌 Status

This project is **actively evolving**.

Expect:
- refactors
- breaking changes
- improved abstractions
- deeper documentation as new stages are implemented

The commit history roughly follows the roadmap above and can be read as a **learning log**.

---

## 🤝 Contributing

This is primarily a **personal learning project**, but contributions are welcome:

- open issues for conceptual discussions
- submit PRs for clarity improvements or bug fixes
- suggest extensions or experiments

If you're also building a framework from scratch, comparisons and discussions are encouraged.

---

## 📣 Build in Public

Progress on this project is documented daily as a **"build your own torch"** series on Twitter/X:

- design decisions
- small code snippets
- lessons learned

The goal is to make this repo useful not just as code, but as a **learning resource**.

---

## 📜 License

MIT License.
