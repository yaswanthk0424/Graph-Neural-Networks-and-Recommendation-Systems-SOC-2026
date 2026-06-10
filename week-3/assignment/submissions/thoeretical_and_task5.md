# Theoretical Proofs

## Task 3: y = x (Linear Function)
### Is an activation function needed?
**No.**

`y = x` is a perfectly linear function. `nn.Linear` already computes:

```
output = weight * x + bias
```

Adding ReLU would introduce non-linearity and break the model's whole purpose to represent a straight line.

### Expected weights and biases

```
weight = 1.0
bias   = 0.0
```

### Defending design

```python
nn.Linear(1, 1)   # 2 parameters total(1 weight, 1 bias)
```

- No hidden layers, no activation function


### Are the weights exactly what you expected?

**Not exactly.** I got `weight ≈ 0.995, bias ≈ 0.005` instead of perfect `1.0, 0.0`.

**Why?**
- Training stopped at finite rounds — not fully converged
- Floating point issues maybe

---

## Task 4: y = x² (Quadratic Function)

### Is an activation function needed?

**Yes.**

`y = x²` is non-linear. A plain `nn.Linear` can only represent `wx + b` — a straight line — which can **never** fit a parabola. ReLU introduces non-linearity, allowing the network to approximate curves.

### Expected weights and biases

There are **no single expected values** here unlike Task 3.

Many combinations of weights can approximate the same parabola — the solution is **not unique**.

### Design Defense

```python
nn.Linear(1, 128)
nn.ReLU()
nn.Linear(128, 64)
nn.ReLU()
nn.Linear(64, 16)
nn.ReLU()
nn.Linear(16, 8)
nn.ReLU()
nn.Linear(8, 1)
```
- ReLU **should not** be used at the end as while training it completely miss out on biases and weights resulting negativity not upgrading the parameters
- More neurons,layers → better approximation but more parameters
- ReLU is required; without it the model collapses back to a linear function

### Are the weights exactly what you expected?

**I cannot expect weights**

Unlike `y = x`, there is no unique weight configuration. The model finds **one valid approximation** among infinitely many. What matters is whether loss is low, not the specific weight values.

---

## Task 5: MSE + tanh vs Cross-Entropy + Softmax

**Cross-Entropy with Softmax performs better for classification.**

- MSE + tanh is used for Regression like for tasks 3,4 type 
- Classification involves fixed possible answers
- outputs are probabilities unlike regression where outputs are proper numbers(in tasks3,4)
