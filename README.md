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
  - `zero_shot`: Boolean flag to enable/disable fine-tuning (note currently only `zero_shot=True` is implemented)
  - `weights`: Path to the pre-trained MLM model (default: HuggingFace GilpinLab/panda_mlm)
  - `normalize`: Boolean flag to normalise the data with the standard routine provided by Panda's repo
  - `context_length`: Maximum length of the context for the foundation model, note that `context_length=-1` corresponds to using the full warm-start data as context, but this needs to be limited in the PDE case due to computational constraints.
  

Example configuration:
```yaml
dataset:
  name: ODE_Lorenz
  pair_id: 1-9 
model:
  name: Panda
  zero_shot: True
  weights: "GilpinLab/panda_mlm"
  normalize: False
  context_length: -1
```

## Requirements

PAnda for CTF relies on Python>=3.10 and the packages listed in `requirements.txt`.