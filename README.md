# Noisy

A diffusion model for image generation with an emphasis on clean and readable
code, an easy to use CLI as well as easy monitoring during training using
[Weights And Biases](https://wandb.ai).

## Requirements

All Python requirements are listed in `requirements.txt`. For training, a CUDA
capable GPU is *strongly* recommended. However, most CLI commands take a
`--device` argument that can be either `cpu` or `cuda`. Thus, all operations can
also be performed on the CPU.

The system was developed and tested on Ubuntu 20.04 LTS. I have not tried it on
Windows (i.e. WSL2) or MAC, but I can not think of an obvious reason why that
should not work.

## Usage and CLI

- `./cli.py init` - Creates a new *project*, based on `/configs/default.yaml`.
- `./cli.py train` - Starts training the previously created project.
- `./cli.py info` - Print some info on the project to stdout.
- `./cli.py sample` - Sample a new image and displays it using matplotlib.
- `./cli.py sample-blend` - Sample a new image and display its intermediate
  stages using matplotlib.

To get more options for each command, add the `--help` flag.

## Concepts

A central conceptual unit of *Noisy* is the checkpoint. A checkpoint is simply a
directory that contains all the files that are associated with a model at a
given point during training. These are:
- the model's state dict (`model.state.pt`),
- the state dict of the Exponential Moving Average (EMA) of the model
  (`ema.state.pt`),
- the state dict of the optimiser (`opt.state.pt`),
- the config (`cfg.yaml`),
- the training context (`ctx.yaml`) and
- information about the performance measurements taken during training
  (`perf.yaml`).

All actions performed with the CLI are performed in the context of a checkpoint.
The latter can be specified with the `--checkpoint` or `-c` argument (e.g.
`./cli.py sample --checkpoint ./my-checkpoint/`). If no checkpoint argument is
passed, the CLI defaults to the most recently active checkpoint. A checkpoint is
activated either with creation (i.e. `./cli.py init`) or with training
(`./cli.py train`).

Another concept is the working directory (also referred to as *workdir* or
*wd*), which is simply a directory containing checkpoints. During training, new
checkpoints are regularly added to the working directory.

To keep working directories in one place, new projects are created inside the
*zoo* by default. The zoo is a directory containing working directories.

## Examples

To initialise a project, train it and finally sample from it, run the following
commands:

```
pip3 install -r requirements.txt
./cli.py init -c configs/default.yaml -w ./zoo/my-project/
./cli.py train  # (interrupt eventually with Ctrl+c to stop training)
./cli.py sample
```

Training can be resumed any time with `./cli.py train -c
./zoo/my-project/.latest`, or simply `./cli.py train` if no new projects were
created in the meantime.

To sample from the most recent checkpoint and show intermediate results, run
`./cli.py sample-blend`.

Note that both `./cli.py sample` and `./cli.py sample-blend` take an `--out`
argument. When supplied, instead of displaying the sampled image, it will be
saved to the specified file (e.g. `./cli.py sample --out my-image.png`).
Similarly, both commands also take a `--number` or `-n` argument which specifies
the number of images to sample. They will then be arranged in a grid.

## Structure

The source tree is structured as follows.

```
noisy/
├── dataset.py      # The dataset and preprocessing pipeline
├── diffusion.py    # Functions specific to the diffusion process
├── __init__.py
├── models          # Collection of model architectures
│   ├── alia.py     # The "Alia" model (147M parameters)
│   ├── base.py     # The abstract base class for the models
│   ├── common.py   # Common building blocks for the models
│   ├── __init__.py
│   ├── lara.py     # The "Lara" model (64M parameters)
│   └── maria.py    # The "Maria" model (123M parameters)
├── perf.py         # Minimal performance measurement tooling
├── training.py     # The training loop and associated functions
├── utils.py        # Utility functions and the `AttrDict`
└── workdir.py      # Functions for dealing with workdirs and checkpoints
```
