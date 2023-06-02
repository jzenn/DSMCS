# Resampling Gradients Vanish in Differentiable Sequential Monte Carlo Samplers [(TinyPaper @ ICLR 2023)](https://openreview.net/forum?id=kBkou5ucR_d)
<div id="top"></div>

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](https://arxiv.org/abs/2304.14390)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  <a href="https://jzenn.github.io" target="_blank">Johannes&nbsp;Zenn</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://robamler.github.io" target="_blank">Robert&nbsp;Bamler</a>


## About The Project
This is the official GitHub repository for our work [Resampling Gradients Vanish in Differentiable Sequential Monte Carlo Samplers](https://openreview.net/forum?id=kBkou5ucR_d) where we propose a differentiable Sequential Monte Carlo Sampler and show that there is no need for differentiating the resampling operation if the effective sample size is maximal (which we find to be the case consistently).

> Annealed Importance Sampling (AIS) moves particles along a Markov chain from a tractable initial distribution to an intractable target distribution. The recently proposed Differentiable AIS (DAIS) (Geffner & Domke, 2021; Zhang et al., 2021) enables efficient optimization of the transition kernels of AIS and of the distributions. However, we observe a low effective sample size in DAIS, indicating degenerate distributions. We thus propose to extend DAIS by a resampling step inspired by Sequential Monte Carlo. Surprisingly, we find empirically—and can explain theoretically—that it is not necessary to differentiate through the resampling step which avoids gradient variance issues observed in similar approaches for Particle Filters (Maddison et al., 2017; Naesseth et al., 2018; Le et al., 2018).

The code base was heavily extended and rewritten based on the DAIS code base by Zhang et al. (2021) which was taken from their [OpenReview submission](https://openreview.net/forum?id=6rqjgrL7Lq).


## Environment: 

- tested with Python 3.10 (lower version might work as well);
- dependencies can be found in `requirements.txt`
- `wandb` is not necessarily required (provide `--wandb` to `train_static_target.py` to enable it)


## Running the Code

Example training command for DAIS baseline:
```bash
python train_static_target.py \
--path <some folder to save experiment results to> \
--log_interval 100 \
--lrate 0.01 \
--batch-size 64 \ 
--epochs 500 \
--max_iterations_per_epoch 10 \ 
--zdim 50 \
--n_particles 32 \ 
--n_transitions 32 \ 
--deltas_nn_output_scale 0.25 \ 
--variational_augmentation UHA
```

The proposed Differentiable Sequential Monte Carlo Sampler can be run by adding
- `--dsmcs --resampling no` (no resampling)
- `--dsmcs --resampling cat` (categorical resampling, without gradient)
- `--dsmcs --resampling gst` (Gapped Gumbel Softmax resampling, with gradient)

Additionally, the "Bernoulli-decision" models are available by `--resampling bern-cat` and `--resampling bern-gst` 


## License
Distributed under the MIT License. See `LICENSE.MIT` for more information.


## Citation:
Following is the Bibtex if you would like to cite our paper :

```bibtex
TODO
```

<p align="right">(<a href="#top">back to top</a>)</p>
