# Dynamic Token Pruning in Vision Transformers

This repository contains the codebase and experimental infrastructure for my Master's thesis on dynamic token pruning and adaptive compute in Vision Transformers.

## Project Structure

```text
dynamic_pruning/
├── configs/        # YAML experiment configurations
├── data/           # datasets, splits, cached files
├── logs/           # runtime logs
├── notes/          # thesis notes and planning material
├── outputs/        # experiment outputs
├── scripts/        # helper scripts for running experiments
├── src/            # source code
│   ├── datasets/
│   ├── models/
│   ├── training/
│   └── utils/
├── README.md
├── requirements.txt
├── requirements_local.txt
└── .gitignore