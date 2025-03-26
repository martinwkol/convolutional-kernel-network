# End-to-End Kernel Learning with Supervised Convolutional Kernel Networks

This repository, **convolutional-kernel-network**, contains an implementation for testing **End-to-End Kernel Learning with Supervised Convolutional Kernel Networks** on the MNIST dataset. The implementation is based on the paper:

**Mairal, J. (2016). End-to-End Kernel Learning with Supervised Convolutional Kernel Networks. ArXiv.Org.**  
[https://doi.org/10.48550/arXiv.1605.06265](https://doi.org/10.48550/arXiv.1605.06265)

## Overview

This project implements and evaluates the **Supervised Convolutional Kernel Networks (SCKN)** approach as described in the paper. It aims to test the efficiency and accuracy of end-to-end kernel learning on the MNIST dataset using Python and `numpy`.

## Repository Structure

- `Mathematical background and experimental results.ipynb`: Jupyter Notebook containing:
  - A detailed explanation of the mathematical background behind SCKN.
  - Experimental results obtained on the MNIST dataset.
- `src/`: Directory containing the Python implementation of the kernel learning approach.

## Installation

### Requirements
- Python 3.x
- `numpy`
- `matplotlib` (for visualizations in the notebook)
- `jupyter` (for running the notebook)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/convolutional-kernel-network.git
   cd convolutional-kernel-network
   ```
2. Install required dependencies:
   ```sh
   pip install numpy matplotlib jupyter
   ```

## Usage

Run the Jupyter Notebook to explore the mathematical background and experimental results:
   ```sh
   jupyter notebook "Mathematical background and experimental results.ipynb"
   ```

## Acknowledgments

- The original research paper by **Julien Mairal** for the theoretical foundation.
- The MNIST dataset for providing a standard benchmark for evaluation.
