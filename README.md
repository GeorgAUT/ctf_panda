# CTF_Panda Model

This directory contains an implementation of the Panda model (Physics-Aware Neural Dynamics) for testing on the CTF for Science Framework, based on the paper "Physics-Aware Neural Dynamics for Learning and Control" (https://arxiv.org/abs/2505.13755).

## Files
- `ctf_panda.py`: Contains the Panda model implementation adapted for the CTF framework.
- `run.py`: Batch runner script for running the model across multiple sub-datasets.
- `config_XX.yaml`: Example configuration file for running the model.

## Usage

To run the Panda model, use the `run.py` script from the **project root** followed by the path to a configuration file. For example:

```bash
python models/ctf_panda/run.py models/ctf_panda/config/config_XX.yaml
```

## Model Description

Panda is a physics-aware neural dynamics model that combines machine learning with physical priors to learn dynamical systems. This implementation uses the MLM-trained model from HuggingFace (https://huggingface.co/GilpinLab/panda_mlm).

### Configuration Structure

Each configuration file must include the following:
- **`dataset`** (required):
  - `name`: The dataset name (e.g., `ODE_Lorenz`, `PDE_KS`).
  - `pair_id`: Specifies sub-datasets to run on. Formats:
    - Single integer: `pair_id: 3`
    - List: `pair_id: [1, 2, 3, 4, 5, 6]`
    - Range string: `pair_id: '1-6'`
    - Omitted or `'all'`: Runs on all sub-datasets.
- **`model`**:
  - `name`: `Panda`
  - `model_path`: Path to the pre-trained MLM model (default: HuggingFace GilpinLab/panda_mlm)
  - `learning_rate`: Learning rate for fine-tuning
  - `num_epochs`: Number of training epochs
  - `batch_size`: Batch size for training
  - `physical_constraints`: Boolean flag to enable/disable physical constraints

Example configuration:
```yaml
dataset:
  name: ODE_Lorenz
  pair_id: 1-9
model:
  name: PAnda
  model_path: "GilpinLab/panda_mlm"
  learning_rate: 1e-4
  num_epochs: 100
  batch_size: 32
  physical_constraints: true
```

## Requirements

PAnda for CTF relies on Python@3.10 and the following packages listed in `requirements.txt`:
- numpy
- torch
- transformers
- accelerate
- datasets
- evaluate
- scipy
- matplotlib
- tqdm
- yaml

## License

This implementation is released under the MIT License.