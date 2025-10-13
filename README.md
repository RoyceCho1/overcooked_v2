# Overcooked V2 Experiments

This repository contains experiments for the Overcooked V2 environment.

## Installation

```bash
conda create -n overcooked_v2 python=3.10
conda activate overcooked_v2
pip install -e JaxMARL
pip install -e experiments
```

It is also possible to build the docker file and run the container.

```bash
make build
make run
```

## Run SP

```bash
python overcooked_v2_experiments/ppo/main.py +experiment=cnn +env=original env.ENV_KWARGS.layout=counter_circuit NUM_SEEDS=10
python overcooked_v2_experiments/ppo/main.py +experiment=rnn-sp +env=grounded_coord_simple NUM_SEEDS=10
```

## Run State Augmentation

```bash
python overcooked_v2_experiments/ppo/main.py +experiment=cnn +env=original env.ENV_KWARGS.layout=counter_circuit NUM_SEEDS=10 NUM_ITERATIONS=10
python overcooked_v2_experiments/ppo/main.py +experiment=rnn-sa +env=grounded_coord_simple NUM_SEEDS=10 NUM_ITERATIONS=10
```

## Run Other Play

```bash
python overcooked_v2_experiments/ppo/main.py +experiment=rnn-op +env=grounded_coord_simple NUM_SEEDS=10
```

## Run FCP

```bash
python overcooked_v2_experiments/main.py +experiment=rnn-fcp +env=grounded_coord_simple NUM_SEEDS=10 +FCP=fcp_populations/grounded_coord_simple
```



