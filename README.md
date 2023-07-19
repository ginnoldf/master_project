# Master Project

### Meeting 05/17/2023

Todos:
- vertical coarse grain and cutoff at some height
  - autocorrelation lengthscale of training data
  - this will lead to a smaller conv kernel
- normalize training data with surface flux 
- take care of w'theta' from subgrid to resolve total value and not only the resolved one (done)
- ask sara for data with larger max x and y
- make code run on casper - create a conda environment

- regularization L1 or L2 on weights
- explore different models (unet)
- use optuna for hyperparameter tuning

- invite Prof. Schär - friday 2.6 might be good, 24.5. as well


### Meeting 05/31/2023

- a lot of work for preprocessing was done again (normalizations, etc. as discussed)
- models still not good enough - as discussed larger datasets and perhaps unet -> I prefer the simple model though
- start meta learning implementation

Todos:
- plotting correct way
- improve baseline (DNN, overfit, perhaps CNN) -> visually good results 
- we use MAML (Juan knows)

### Meeting 07/03/2023

- baseline should be good now -> results are visually good
- maml implementation is done - it works for smaller datasets
  - it is not better than training with all datasets
  - I do not get the hyperparameters right
- leave out is interesting

Todos:
- meeting with Juan to discuss maml setup
- different kind of data - start with Laure?
- prepare meeting with Prof. Schär for next time
- transfer learning approach, just unfreeze one layer at the end or the first 2 cnn layers

### Meeting 07/20/2023


