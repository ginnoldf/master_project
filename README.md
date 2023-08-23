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

- maml: only remaining problem is MSE for the base dataset
- transfer learning: a little worse, as expected
- experiments take long and finetuning is frustrating
- what should be my priorities in the next 2.5 months, what outcome is expected (good reusable, commented and in
  README.md files explained code, easily reproducable results, many prepared experiments or just the main ones, 
  collection of important tables/graphs, presentation for the group, thesis paper)

notes:
- latent space in MAML
- paper on transfer learning approach
- https://arxiv.org/pdf/1703.03400.pdf
- https://academic.oup.com/pnasnexus/article-abstract/2/3/pgad015/6998042
- https://www.sciencedirect.com/science/article/pii/S0021999122001528?casa_token=qSelgmHslhoAAAAA:blCti2x6d1pO2-INndWD7joeaAOuGz33nvwJwW3HsLyB59gGGedDCB8LZlRmv9epBe70im6Pqg
- meeting with Christoph
- meeting with Laure
- evaluate the same samples for all evaluations (about 10) 


### Meeting 08/23/2023

- update on ocean data progress
- plan for last weeks:
  - https://docs.google.com/spreadsheets/d/1R2dq5GlGxvZhh2bK7Bx9XyiHy_FN2tExlXzBYUqkMF0
  - when I write the first draft, is there somebody who can go over it?
  - what meetings do we need? one with Laure when I have ocean results and a final one?
  - where should my code end up - in a Leap or M2LInES repo?
- what is expected from my thesis document? present structure so far
- should we continue with transfer learning somehow?
