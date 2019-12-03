# ap-perf-py

This repository provides a Python interface to the [AdversarialPrediction.jl](https://github.com/rizalzaf/AdversarialPrediction.jl). Using this repository, we can integrate the adversarial prediction formulation over generic performance metrics into differentiable learning pipeline of PyTorch. The implementation is based on the paper: ["AP-Perf: Incorporating Generic Performance Metrics in Differentiable Learning"](http://arxiv.org/abs/1912.00965) by [Rizal Fathony](http://rizal.fathony.com) and [Zico Kolter](http://zicokolter.com). 
Please check the tutorial of the [AdversarialPrediction.jl](https://github.com/rizalzaf/AdversarialPrediction.jl) first, before continuing this tutorial.

**Note: A pure Python implementation of the AdversarialPrediction.jl is coming soon.**

## Overview

`ap-perf-py` enables easy integration custom performance metrics (including non-decomposable metrics) into  PyTorch learning pipelines. It currently supports metrics that are defined over binary classification problems.
Below is a code example for incorporating the F-2 score metric into a neural network training pipeline of PyTorch. 

```python
from metric import Metric
from layer import MetricLayer

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze() 

# metric definition
f2_str = """ 
@metric F2Score
function define(::Type{F2Score}, C::ConfusionMatrix)
    return ((1 + 2^2) * C.tp) / (2^2 * C.ap + C.pp)  
end   
f2_score = F2Score() 
"""

# instantiate metric object
f2_score = Metric(f2_str)
f2_score.initialize()
f2_score.special_case_positive()

# create a model and criterion layer
model = Net().to(device)
criterion = MetricLayer(f2_score).to(device)

# forward pass
optimizer.zero_grad()
output = model(inputs)
objective = criterion(output, labels)

# backward pass
objective.backward()
optimizer.step()
```

Note that we can easily write the definition of the F2-score from the entities in the confusion matrix, and then incorporate it into PyTorch training pilepines using `MetricLayer`. This is a very simple modification from the standard binary learning setting that uses the binary cross-entropy as the objective, which only differ in the way we construct `criterion`:
```python
criterion = nn.BCEWithLogitsLoss().to(device)
```

## Installation

`ap-perf-py` relies on [PyJulia](https://pyjulia.readthedocs.io/en/latest/) to call Julia code from Python. Therefore, to install `ap-perf-py`, we need to have a Julia installation available in the PATH system variable, and `PyJulia` installed in the Python environment. Please visit [Julia's download page](https://julialang.org/downloads/) to find instructions on installing Julia, and [PyJulia installation instructions](https://pyjulia.readthedocs.io/en/latest/installation.html) to setup `PyJulia`. Please also check some of the [known issues](https://pyjulia.readthedocs.io/en/latest/troubleshooting.html) in installing PyJulia.

To use this repository, simply download it and copy it into a local folder. Given we have installed Julia and PyJulia, we can install AdversarialPrediction.jl and its dependency packages in Julia from Python using the following code:
```python
import install
install.install_with_gurobi()
```
if we want to use `Gurobi` as the LP solver or
```python
import install
install.install_no_gurobi()
```
if we do not have access to Gurobi. For `install_with_gurobi()`, Gurobi executable must be available in the system PATH.
Note that we have to run Python from the directory of this project to run the codes above. 


## Defining Performance Metrics

`ap-perf-py` provides a Python class `Metric` to construct performance metrics from the entities in the confusion matrix. However, since the Julia compiler needs to have access to the definition of `define` and `constraint` functions, we cannot implement them in Python. Instead, we need to write the definitions as a string object in Python, and then pass it to `Metric` class, which then be directed to Julia runtime. 
Below is an example of defining the F-2 score with the F-beta class.  

```python
from metric import Metric
Metric.set_solver("GUROBI")

## Precision given recall
fbeta_str = """ 
@metric FBeta beta
function define(::Type{FBeta}, C::ConfusionMatrix, beta)
    return ((1 + beta^2) * C.tp) / (beta^2 * C.ap + C.pp)  
end   
"""

f2_score = Metric(fbeta_str)
f2_score.initialize(2.0)
f2_score.special_case_positive()
```

The string variable for the definition must contain `@metric` statement, `define` function, and `constraint` function (if needed). 
To instantiate a metric object, we can write pythonic codes, as shown above. If a metric has parameters, we need to provide the parameters when we call the `initialize` method.
`ap-perf-py` also provides pythonic methods for modifying and querying the metric. This includes: `special_case_positive`, `special_case_negative`, `cs_special_case_positive`, `cs_special_case_negative`, `compute_metric`, `compute_constraints`, and `objective`.
Below is an example for the case of metric with constraints:
```python
from metric import Metric
Metric.set_solver("GUROBI")

## Precision given recall
precrec_str = """
@metric PrecRec th
function define(::Type{PrecRec}, C::ConfusionMatrix, th)
    return C.tp / C.pp
end   
function constraint(::Type{PrecRec}, C::ConfusionMatrix, th)
    return C.tp / C.ap >= th
end 
"""

precrec80 = Metric(precrec_str)
precrec80.initialize(0.8)
precrec80.special_case_positive()
precrec80.cs_special_case_positive(True)
```

Note that if we want to use `Gurobi` as the LP solver, we need to write `Metric.set_solver("GUROBI")` before initializing the metric. Otherwise, `ECOS` will be used as the solver.

## Computing the Values of the Metric

Given we have a prediction for each sample `yhat` and the true label `y`, we can call the function `compute_metric` to compute the value of the metric. Both `yhat` and `y` are vectors containing 0 or 1. 
```python
>>> f1_score.compute_metric(yhat, y)
0.8
```  

For a metric with constraints, we can call the method `compute_constraints` to compute the value of every  metric in the constraints. For example:
```python
>>> precrec80.compute_constraints(yhat, y)
array([0.7], dtype=float32)
```

## Integration with PyTorch

The class `MetricLayer` provides an easy way to incorporate the performance metric into PyTorch learning pipelines. The `MetricLayer` inherits PyTorch's `nn.Module` class, which serves as the objective or the last layer in the neural network learning pipeline. To use `MetricLayer`, simply instantiate it with the defined metric as an input, and then use it as the objective function in the training pipeline.

```python
from layer import MetricLayer

criterion = MetricLayer(f2_score).to(device)

# forward pass
optimizer.zero_grad()
output = model(inputs)
objective = criterion(output, labels)

# backward pass
objective.backward()
optimizer.step()
```


## Code Example

We provide a simple example for integrating  `ap-perf-py` into PyTorch code for a small MLP network in `simple.py` file included in this repository. Please also check [AP-examples](https://github.com/rizalzaf/AP-examples) for more examples in Julia. 

## Citation

Please cite the following paper if you use this repository for your research.

```
@article{ap-perf,
  title={AP-Perf: Incorporating Generic Performance Metrics in Differentiable Learning},
  author={Fathony, Rizal and Kolter, Zico},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2019}
}
```

