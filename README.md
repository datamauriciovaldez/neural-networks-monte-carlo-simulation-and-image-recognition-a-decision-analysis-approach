# Methods of Decision Analysis – Final Project

Authors: Javier Gonzalo & Mauricio Valdez (BIDA, 2nd year).

Three tasks, each in a Jupyter notebook:

| Task | Notebook | Topic |
|------|----------|-------|
| 1 | `task1_montecarlo_simulation.ipynb` | Monte Carlo simulation (N = 100,000) comparing In-house, Outsource and Chatbot support strategies on profit and risk. Plots in `Task 1 plots/`. |
| 2 | `mauri_Final_2 copia.ipynb` | Neural network ([4,5,2]) built from scratch with NumPy to predict Titanic survival (`Titanic_Dataset.csv`); compared against a linear model. Explores features, nodes and batch sizes. |
| 3 | `Final_3_torch copia 3.ipynb`, `Final_3_torch copia 31.ipynb` | MNIST digit recognition (784-N-10 network, 60,000 examples), NumPy version and a PyTorch version. |

## Requirements
Python 3, numpy, pandas, matplotlib, seaborn, scikit-learn, torch, Jupyter.

## Notes
- Task 3 loads MNIST from a local gzip/pickle file (`mnist.pkl.gz`) that is **not** included in the repo; download it separately and place it next to the notebooks.
