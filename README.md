# ssgp
A project implementing sparse spectrum Gaussian process regression using PyTorch and quasi-random Fourier features generated using Halton sequences.

## Requirements
  - PyTorch >= 1.1 (<https://pytorch.org/>)
  - ghalton >= 0.6.1 (<https://github.com/fmder/ghalton>)

### Optional
  - NLopt (<https://pypi.org/project/nlopt/>): for hyper-parameters tuning
  - Jupyter Notebook (<https://jupyter.org/>): to run example notebook
  - Matplotlib (<https://matplotlib.org/>): to plot in example notebook

## Installation
  1. Clone git repository
  ```
  git clone git@bitbucket.org:rafaol/ssgp.git
  ```
  2. Switch to repository's root directory:
  ```
  cd ssgp/
  ```
  3. Install dependencies:
  ```
  pip install -r requirements.txt
  ```
  4. Install package in editable mode (-e option) to keep track of updates: 
  ```
  pip install -e .
  ```
