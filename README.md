# Transfer Learning Project

This repository contains the official implementation of Julian van Toledo's MSc Project, focusing on transfer learning in survival analysis.

## Setup and Installation

### Prerequisites
- Python 3.7.12
- R version 4.2.0
- Docker

### Installation
1. **Docker Setup:**
   - Run the following commands to set up the environment using Docker:
     ```bash
     docker build -t transferlearning .
     docker run -it transferlearning
     ```
   - This will install all necessary dependencies and create the right environment.

2. **Running the Script:**
   - After setting up the Docker environment, execute the main script:
     ```bash
     python transferlearning.py
     ```

## Files and Scripts

### Main Script
- **`transferlearning.py`**: The primary script implementing the transfer learning model.

### Configuration and Other Files
- **`Dockerfile`**: Docker configuration for setting up the environment.
- **`requirements.txt`**: Lists all Python dependencies.
- **`.gitignore`**: Specifies files to ignore in version control.
- **`transferlearning.yaml`**: Configuration file for the transfer learning process.

### Archive
These scripts are not necessary for the main transfer learning procedure and are archived:
- `benchmark.py`
- `cox_nnet_v2.py`
- `hyperparam_opt.py`
- `nnet_survival.py`
- `pre_dataset.py`
- `Preprocessing_icgc.R`
- `preprocessingscript.R`
- `TensorBoardNotebook.ipynb`
- `gpu_test.ipynb`
- `nnetsurvivaltransfer.ipynb`
- `transferlearning_1.py`
- `transferlearning_2.py`
- `transferlearning_3.py`
- `transferlearning_4.py`

## Authors
- Julian van Toledo

## License
- [MIT License](LICENSE)
