# MLP Backpropagation (Java)

[![Live Visualizer](https://img.shields.io/badge/🌐_Interactive_Demo-Visit_Web_App-007ACC?style=for-the-badge&logo=vercel&logoColor=white)](https://java-mlp-backpropagation-neural-net.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

A from-scratch implementation of a single-hidden-layer Multilayer Perceptron (MLP), trained
with online (per-sample) backpropagation and sigmoid activations. No external ML libraries —
every matrix operation, activation function, and gradient update is implemented by hand in
plain Java, which makes this a good reference for understanding how backpropagation actually
works under the hood.

🚀 **Try the interactive web visualization by** [clicking here](https://java-mlp-backpropagation-neural-net.vercel.app/)**.**

Originally written in 2019 as a coursework project (for University of São Paulo - USP - Academics' purposes); hardened and verified in 2026 (see
[Project History](#project-history) for what changed).

## Features

- Configurable network: any number of input attributes, hidden neurons, and output classes.
- Full feedforward + backpropagation implemented from first principles (matrix multiplication,
  Hadamard product, transpose, sigmoid derivative — all hand-rolled in `Matrix.java`).
- Automatic learning-rate decay for more stable convergence.
- Stagnation detection with automatic random-restart recovery, so a single unlucky random
  weight initialization can't stall training indefinitely.
- A hard epoch cap as a safety net against non-convergent configurations.
- An optional verbose debug mode for inspecting every intermediate matrix during training.

## Project Structure

| File                | Responsibility                                                                  |
|---------------------|---------------------------------------------------------------------------------|
| `Main.java`         | Entry point. Reads hyperparameters and file paths, drives the training loop, then runs the trained network against a test set. |
| `RedeNeural.java`    | The network itself: weights, biases, hyperparameters, and their getters/setters. |
| `Operations.java`    | Feedforward pass, backpropagation, and the sigmoid activation function + its derivative. |
| `Matrix.java`        | All matrix math: multiply, add, subtract, transpose, Hadamard product, scalar multiply, random initialization, printing. |
| `LeitorCSV.java`     | Reads a CSV dataset into input/target matrices. |

All five files belong to the `mlp_backpropagation` package.

## Requirements

- JDK 8 or newer (tested with OpenJDK 21).

## Getting Started

### 1. Folder layout

Every file declares `package mlp_backpropagation;`, so they must live inside a folder
**named exactly `mlp_backpropagation`**:

```
your-project/
└── mlp_backpropagation/
    ├── Main.java
    ├── RedeNeural.java
    ├── Operations.java
    ├── Matrix.java
    └── LeitorCSV.java
```

> **"Incorrect Package" error?** This means the folder name doesn't match the package
> declaration — usually because the files were dropped directly into a folder instead of
> a subfolder called `mlp_backpropagation`. Fix the folder structure above, or strip the
> `package mlp_backpropagation;` line from all five files if you'd rather not deal with the
> folder requirement at all.

### 2. Compile

From the parent directory (one level above `mlp_backpropagation/`):

```bash
javac mlp_backpropagation/*.java
```

### 3. Run

```bash
java -cp . mlp_backpropagation.Main
```

The program will prompt you interactively, in this order:

| Prompt                                    | Meaning                                             | Example |
|-------------------------------------------|-----------------------------------------------------|---------|
| `Caminho do dataset de treino:`            | Path to the training CSV file                       | `dados/treino.csv` |
| `Numero de neuronios na camada escondida:` | Number of hidden-layer neurons                       | `10` |
| `Numero de atributos de camada entrada:`   | Number of input attributes per row                   | `63` |
| `Numero de saidas possiveis:`              | Number of output classes                              | `7` |
| `Valor da taxa de aprendizado inicial:`    | Starting learning rate (0–1). Accepts `.` or `,`      | `0.3` |
| `Buscar valor inferior a qual erro médio:` | Target average error to stop training. Accepts `.` or `,` | `0.1` |
| `PARA INICIAR OS TESTES FORNECA CAMINHO...`| Path to the test CSV file (asked once training finishes) | `dados/teste.csv` |

The same attribute/output counts you provide are used for both the training and test files,
so both datasets must share the same column layout.

## Input Data Format

- Plain CSV, comma-separated, **no header row**.
- Each row is: `N` attribute values, followed by exactly one class label as the **last**
  column.
- Attribute values are restricted to exactly `0`, `1`, `1.5`, or any negative number
  (e.g. `-1`, `-0.5`). Any other numeric value (like `0.7` or `2`) will be misread as a class
  label, not an attribute — the parser was written for a specific encoded dataset, not
  general-purpose numeric CSVs.
- Supported class labels are `A`, `B`, `C`, `D`, `E`, `J`, `K` (mapped internally to positions
  0–6). Any other label is silently ignored — the row's target stays all zeros.

Example row (10 attributes + label `A`):

```
-1,0,1,1.5,0,-1,1,0,0,1.5,A
```

## Configuration / Hyperparameters

- **Hidden neurons**: more neurons = more capacity, but also more parameters to fit with
  whatever training data you provide. There's no fixed rule — start small and increase if the
  network can't drive the training error down.
- **Learning rate**: the starting value you provide decays automatically each epoch (see
  `LR_DECAY` in `Main.java`), so it's fine to start relatively high (e.g. `0.3`).
- **Target error**: training stops once the average per-sample error drops below this value.
  Very tight thresholds may need many epochs — or may never be reachable for noisy data with
  a small network, in which case training stops at the epoch cap and reports the final error
  it reached instead of hanging forever.
- **Restarts**: if training stalls (no meaningful error improvement for 300 epochs), the
  network automatically reinitializes with fresh random weights and tries again, up to 20
  times. This is normal, expected behavior, not an error — it's printed to the console when
  it happens.

## How It Works

The network has exactly one hidden layer:

```
input (N attributes) → hidden layer (sigmoid) → output layer (sigmoid)
```

Training is **online** (per-sample) gradient descent: for every single training row, the
network does a full forward pass, computes the error against the target, and immediately
backpropagates that error to update every weight and bias — rather than accumulating
gradients across a batch. One full pass over the dataset is one "epoch"; the average error
per sample is printed at the end of every epoch (`CUSTO MÉDIO`) so you can watch training
progress even without verbose mode enabled.

## Debugging

Set `LOG_DETALHADO = true` at the top of `Main.java` to print every intermediate matrix
(weights, biases, activations, error terms) for every single training sample. This is very
verbose — expect large console output — so it's off by default and meant purely for
inspecting the internals, not for normal runs.

## Known Limitations

- The CSV parser's value/format rules (see [Input Data Format](#input-data-format)) are
  specific to the original dataset this was built for, not a general-purpose CSV reader.
- Only seven hardcoded class labels (`A–E`, `J`, `K`) are recognized.
- The class label must be the last column in each row; the parser does not correctly handle
  labels placed elsewhere in a row.
- No built-in regularization — on small or noisy datasets the network can fit the training
  data very well while generalizing less well to unseen data. This is a property of the
  training approach (no dropout, weight decay, or validation-based early stopping), not a
  bug.

## Project History

This project was originally written in 2019. During a 2026 review, the following issues were
found and fixed:

- Several matrix/array dimension mismatches (`RedeNeural`, `LeitorCSV`) that caused crashes
  whenever the hidden layer size differed from the output size, or the dataset had more rows
  than attributes.
- A `Scanner` bug in `Main.java` where a second `Scanner` opened on `System.in` for the test
  phase silently discarded buffered input.
- The test phase was evaluating the training data instead of the test data.
- A locale-dependent input bug where the decimal separator prompt only worked correctly on
  certain regional settings.
- Two correctness bugs in the backpropagation math (`Operations.java`): the hidden layer's
  error term was missing the output layer's sigmoid-derivative factor, and it was propagated
  through weights that had already been updated for the current step instead of the weights
  used during the forward pass.
- A performance issue where full weight/bias matrices were printed to the console for every
  single training sample, making realistic training runs extremely slow — now gated behind
  the `LOG_DETALHADO` flag, off by default.
- A random-initialization bug in `Matrix.Randomize` where weight *signs* followed a fixed,
  repeating pattern rather than being genuinely random, which could leave the network unable
  to learn from certain inputs regardless of training time.
- Added a training epoch cap and automatic stagnation-detection restart, since even with
  correct math and initialization, a small percentage of random initializations can still
  land in a poor starting point that fixed-rate gradient descent can't escape on its own.

All fixes were verified by compiling and running the code directly, including repeated trials
on both the original dataset shape and a minimal XOR test case (a classic check for whether a
two-layer network's backpropagation is implemented correctly, since XOR cannot be solved by a
single layer).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Massao Tani
