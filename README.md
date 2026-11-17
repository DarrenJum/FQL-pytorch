# Flow Q-Learning (PyTorch)

[![Paper](https://img.shields.io/badge/NeurIPS_2026-arXiv:2610.38238-b31b1b.svg)](https://arxiv.org/abs/2610.38238)
[![Python 3.9+](https://img.shields.io/badge/Python-3.8.5-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)


## Overview

An unofficial PyTorch implementation of **[Flow Q-learning (FQL)](https://github.com/seohongpark/fql)**

## Environment

1. Install [MuJoCo version 2.1](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0) at ~/.mujoco/mujoco210
2. Create a conda environment
```
conda env create -f conda_env.yml
conda activate fql
```
3. Install [D4RL](https://github.com/Farama-Foundation/D4RL/tree/4aff6f8c46f62f9a57f79caa9287efefa45b6688)

## Usage

### Offline RL

Run the following command to train offline RL on D4RL.

```
python main.py --config configs/offline/hopper-medium.yml
python main.py --config configs/offline/antmaze-large-diverse.yml
```

You can also specify the random seed:

```
python main.py --config configs/offline/hopper-medium.yml --seed <seed>
```

#### Logging

This codebase uses tensorboard. You can view saved runs with:

```
tensorboard --logdir <run_dir>
```

### Online Fine-tuning

Run the following command to online fine-tune on AntMaze with pretrained offline agents.

```
python main_finetune.py --config configs/online_finetune/antmaze-large-diverse.yml
```

You can also specify the random seed offline models:

```
python main_finetune.py --config configs/online_finetune/antmaze-large-diverse.yml --seed <seed> --pretrain_model <pretrain_model/>
```
