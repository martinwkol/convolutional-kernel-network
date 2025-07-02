# End-to-End Kernel Learning with Convolutional Kernel Networks

This project implements and analyzes the technique described in:

> **Julien Mairal** (2016).  
> _End-to-End Kernel Learning with Supervised Convolutional Kernel Networks_.  
> [arXiv:1605.06265](https://doi.org/10.48550/arXiv.1605.06265)

The implementation is focused on understanding the core ideas of **Convolutional Kernel Networks (CKNs)** and reproducing experimental results on the MNIST dataset.

---

## 📂 Project Structure

```

.
├── Mathematical background and experimental results.ipynb  # Main analysis
├── Mathematical background and experimental results.pdf    # PDF export of the notebook
├── mnist/                                                  # MNIST dataset files
├── README.md
└── src/
   └── *.py    # Core implementation and training script

````

- The Jupyter notebook contains:
  - A detailed mathematical explanation of CKNs.
  - Experiments and visualizations on MNIST.
- The Python code in `src/` provides a working example of training a network on MNIST using the method.

---

## ⚙️ Installation

Requires Python 3.x. Install dependencies:

```bash
pip install numpy matplotlib
````

For notebook usage:

```bash
pip install jupyter
```

---

## 🚀 Running the Code

### 🧪 To run the notebook:

```bash
jupyter notebook "Mathematical background and experimental results.ipynb"
```

This contains the full explanation and results.

---

### 🏁 To train an example network:

You can run the training script as an example of using the method:

```bash
python src/train_mnist.py -f cknet_mnist_model.pkl -e 10
```

Optional arguments:

* `-m`: path to MNIST directory (default: `mnist/`)
* `-e`: number of epochs
* `-nt`: number of test samples
* `-et`: epochs between tests
* `--initial-test`: run a test before training starts

---

## 📌 Notes

* This project is **not packaged as a reusable library**.
* The code is structured for educational and analytical purposes.
* You are free to explore or adapt the scripts in `src/` as needed.

---

## 🙏 Acknowledgments

* Thanks to Julien Mairal for the original paper and technique.
* The MNIST dataset is used as a benchmark for evaluation.


---

## 📖 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute it as you wish.

