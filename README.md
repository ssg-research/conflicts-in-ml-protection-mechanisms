# Conflicting Interactions Among Protection Mechanisms for Machine Learning Models

This is a code repo for [Conflicting Interactions Among Protection Mechanisms for Machine Learning Models](https://arxiv.org/abs/2207.01991); to appear in AAAI 2023.

## Requirements

For autoalloc of GPUs: you need jc and nvidia-smi; by default, it's disabled.

This codebase uses wandb as the logging backend.

You need __conda__. Create a virtual environment and install requirements:

```bash
conda env create -f environment.yaml
```

To activate:

```bash
conda activate ml-conf-interests
```

To update the env:

```bash
conda env update --name ml-conf-interests --file environment.yaml
```

or

```bash
conda activate ml-conf-interests
conda env update --file environment.yaml
```

## Run

**Disclaimer:** Run all experiments from the **$ROOT** of the project.

Running DP + watermarking:

```bash
python3 -u -m src.main task=dp-wm
```

Running adversarial training + watermarking:

```bash
python3 -u -m src.main task=adv-wm
```

Running adversarial training + dp + watermarking:

```bash
python3 -u -m src.main task=adv-dp-wm
```

You also need to specify data/wm/model that you want to train with, e.g.:

```bash
python3 -u -m src.main task=dp-wm learner=mnist
```

Invoke `--help` for more info.

To run experiments with different hyperparams, you can overwrite them from the CLI or create new config files. See [hydra's documentation](hydra.cc) for more info.

## DI

Train models with this script, and use them from the [official DI repo](https://github.com/cleverhans-lab/dataset-inference)

## Radioactive data

[This is the official repo](https://github.com/facebookresearch/radioactive_data).
However, we use the modified version from [this work](https://arxiv.org/abs/2202.12506) which extends the original code to other datasets.
That code is available upon request from the authors.
