# SALSA

**S**equential **A**daptive **L**earning **S**ampling **A**rchitecture

A lightweight, modular Python framework for **Active Learning** (including 
multi-fidelity setups) and **Bayesian Optimization** experiments.

SALSA provides clean abstractions for the core components of a BO/AL loop — 
surrogate models, acquisition functions, samplers, and objective functions — 
so you can swap them in and out without rewriting your experiment code.

## Origin

SALSA was born during the *ABS26 - Applied Bayesian Statistics Summer School 
2026*, held at Villa del Grumello (Como, Italy). I became interested in 
surrogate modeling through Simon Mak's (Duke University) lectures on Gaussian 
Processes, multi-fidelity methods, Active Learning and Bayesian Optimization. 
I kept studying the topic afterwards and built SALSA to get hands-on experience 
with the implementation side of the BO/AL world.

## Features

- Modular surrogate models (Gaussian Processes, ...)
- Standard acquisition functions: Expected Improvement (EI), 
  Probability of Improvement (PI), Upper Confidence Bound (UCB)
- Multi-fidelity optimization support
- Active learning samplers
- Ready-to-use benchmark problems (Branin, ...) as reference implementations
- Jupyter notebook tutorials

## Installation

```bash
git clone https://github.com/morettitommaso/SALSA.git
cd SALSA
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Defining your own problem

SALSA is designed to be extended. To use it on your own objective function,
subclass `Problem` and implement `evaluate`:

```python
import numpy as np
from salsa.problems.base import Problem


class MyProblem(Problem):

    def __init__(self):
        self.name = "MyProblem"
        self.dimension = 2
        self.bounds = np.array([
            [-5, 10],
            [0, 15]
        ])
        self.min = 0.0
        self.max = 1.0

    def evaluate(self, X):
        # X is a (n_points, dimension) array
        return (X ** 2).sum(axis=1)
```


For `multi-fidelity` problems, subclass `MultiFidelityProblem` and implement
`evaluate_high` and `evaluate_low`:

```python
from salsa.problems.base import MultiFidelityProblem


class MyMultiFidelityProblem(MultiFidelityProblem):

    def __init__(self):
        self.name = "MyMultiFidelityProblem"
        self.dimension = 2
        self.bounds = np.array([
            [-5, 10],
            [0, 15]
        ])

    def evaluate_high(self, X):
        # expensive, high-fidelity evaluation
        ...

    def evaluate_low(self, X):
        # cheap, low-fidelity approximation
        ...
```


## Project structure

```text
src/
├── acquisition/     # EI, PI, UCB, ...
├── surrogates/      # GP models, ...
├── sampler/         # sampling strategies
├── problems/        # benchmark problems
├── evaluation/      # metrics and experiment runners
└── visualization/   # plotting utilities
```

## Tutorials

- `notebooks/01_getting_started.ipynb` | basic BO loop  
- `notebooks/02_multi_fidelity_learning.ipynb` | multi-fidelity setup  
- `notebooks/03_bayesian_optimization.ipynb` | full experiment  


## Possible future implementations

- Cost-aware Active Learning  
- Batch Active Learning  
- Active Learning for classification problems  
- Multi-fidelity Bayesian Optimization  


## References

[1] Gramacy, R.B. (2020). Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences (1st ed.). Chapman and Hall/CRC. https://doi.org/10.1201/9780367815493

[2] MacKay, D. J. C. (1992). Information-based objective functions for active data selection. Neural Computation, 4(4), 590–604.

[3] Cohn, D. A., Ghahramani, Z., & Jordan, M. I. (1996). Active learning with statistical models. Journal of Artificial Intelligence Research, 4, 129–145.

[5] Kennedy, Marc C and O’Hagan, Anthony. Predicting the output from a complex computer code when fast approximations are available. Biometrika, 87(1):1–13, 2000

[6] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. Journal of Global Optimization, 13(4), 455–492.

[7] Multi-fidelity Gaussian process surrogate modeling for regression problems in physics


## License

MIT - see [LICENSE](https://github.com/morettitommaso/SALSA/blob/main/LICENSE).