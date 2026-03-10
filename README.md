# Compute-Aware Vision Transformers – Thesis Repository

This repository contains the experimental framework for a Master's thesis on **compute-aware Vision Transformers (ViTs)**. The goal of the project is to study methods for improving the **accuracy–efficiency trade-off** in Vision Transformers by dynamically controlling computation during inference or training.

The repository provides a **reproducible experiment infrastructure** that supports:

- configuration-driven experiments
- reproducible training pipelines
- local development and testing
- GPU execution on the **ULHPC cluster**
- structured logging and experiment outputs

The framework will later be extended with **dynamic token pruning and compute-aware strategies for Vision Transformers**.

---

# Repository Structure

```
compute-aware-vit-thesis
│
├── configs/                  # Experiment configuration files
│   ├── base.yaml
│   ├── debug.yaml
│   └── baseline_dense.yaml
│
├── src/                      # Training pipeline and utilities
│   ├── train.py
│   ├── datasets/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── scripts/                  # Execution scripts
│   ├── run_local.ps1         # Local execution (Windows)
│   └── run_hpc.sh            # SLURM execution (ULHPC)
│
├── outputs/                  # Experiment outputs
├── logs/                     # SLURM job logs
│
├── requirements.txt
├── requirements_local.txt
├── requirements_ulhpc_model.txt
│
└── README.md
```

---

# Experiment Configuration

Experiments are defined using **YAML configuration files**.

Example configuration:

```yaml
experiment_name: debug_run

model:
  name: vit_base_patch16_224
  num_classes: 10

training:
  batch_size: 4
  epochs: 1
  learning_rate: 0.0001

data:
  dataset_name: cifar10
```

Using configuration files allows experiments to be reproduced easily and avoids modifying training code for different runs.

---

# Local Development Environment

Local development is used for:

- debugging the training pipeline
- testing configuration files
- verifying the code structure

Create the environment (example with Conda):

```bash
conda create -n dynamic_pruning python=3.10
conda activate dynamic_pruning
```

Install dependencies:

```bash
pip install -r requirements_local.txt
```

Run a local test experiment:

```bash
python src/train.py --config configs/debug.yaml
```

This will execute a short test run and create a timestamped output folder inside `outputs/`.

---

# ULHPC Environment

Experiments requiring GPUs are executed on the **ULHPC cluster**.

A dedicated Python environment was created on ULHPC to ensure compatibility with GPU libraries.

Activate the environment:

```bash
source ~/dynamic_pruning_model_env/bin/activate
```

The environment includes:

- PyTorch
- torchvision
- timm
- transformers
- datasets
- wandb
- supporting ML libraries

All dependencies are stored in:

```
requirements_ulhpc_model.txt
```

---

# Running Experiments on ULHPC

Experiments are submitted using SLURM.

Submit a job:

```bash
sbatch scripts/run_hpc.sh configs/debug.yaml
```

or

```bash
sbatch scripts/run_hpc.sh configs/baseline_dense.yaml
```

The script will:

1. activate the ULHPC environment  
2. create log and output folders  
3. run the training pipeline  

---

# Logs and Outputs

SLURM job logs are saved in:

```
logs/
```

Standard output:

```
logs/<jobname>_<jobid>.out
```

Error logs:

```
logs/<jobname>_<jobid>.err
```

Experiment results are stored in:

```
outputs/<experiment_name_timestamp>/
```

Each run gets its own folder to ensure experiments do not overwrite each other.

---

# Reproducibility

The project ensures reproducibility through:

- fixed random seeds
- configuration-driven experiments
- dependency freezing
- version-controlled code

Dependency files:

```
requirements_local.txt
requirements_ulhpc_model.txt
```

These files allow environments to be recreated exactly.

---

# Current Status

The current stage of the project includes:

- experiment infrastructure
- configuration system
- training pipeline skeleton
- ULHPC GPU integration
- SLURM batch execution

The next development stages will include:

- implementing a baseline Vision Transformer model
- integrating datasets and training loops
- implementing dynamic token pruning methods
- evaluating compute-accuracy trade-offs

---

# Thesis Objective

The objective of this research is to investigate **compute-efficient Vision Transformer architectures** by dynamically adjusting the number of processed tokens during inference or training.

Future work will explore:

- token pruning strategies
- adaptive token allocation
- compute-aware attention mechanisms
- efficiency-accuracy trade-offs in Vision Transformers

---

# Author

**Arooba Arshad**  
Master's Thesis – Computer Science  
University of Luxembourg