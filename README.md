# Machine Learning Classification & Pattern Recognition Suite

This repository contains a comprehensive suite of machine learning algorithms and pattern recognition techniques implemented in Python. The project is divided into four distinct parts, ranging from fundamental statistical estimation to high-dimensional classification using Support Vector Machines (SVM).

##  Project Structure & Datasets

The project is structured in a single Jupyter Notebook, divided into four main sections. All required datasets are stored in the `data/` directory. Each section processes its respective dataset to evaluate different algorithms.

### Part A: Maximum Likelihood Estimation (MLE)
* **Task:** Estimating the parameters ($\mu$ and $\Sigma$) of Multivariate Gaussian Distributions for 3 distinct classes in a 2D feature space.
* **Highlights:** Vectorized calculation of covariance matrices and 3D visualization of probability density functions.

### Part B: Parzen Windows Density Estimation
* **Task:** Estimating the probability density function of 1D data using non-parametric methods.
* **Highlights:** Implemented both **Gaussian** and **Hypercube** kernels. Conducted hyperparameter tuning to find the optimal window size ($h$) that minimizes the Mean Squared Error (MSE) compared to the true distribution.

### Part C: K-Nearest Neighbors (KNN) from Scratch
* **Task:** Binary classification using a custom-built KNN classifier.
* Highlights:** Implemented two variants from scratch: **Voting KNN** and **Density Estimation KNN**. Resolved tie-breaking scenarios using cumulative distance metrics and plotted complex, non-linear decision boundaries.

### Part D: High-Dimensional Classification & Feature Analysis
* **Task:** Classifying a highly complex dataset comprising 8,743 samples and 224 features into 5 classes.
* **Highlights:** * Evaluated multiple models (SVC, KNN, MLP, Random Forest) and selected **SVC with RBF kernel** after extensive hyperparameter tuning ($C=2.7$, gamma='scale').
  * Performed deep feature analysis using **Cohen's d** to quantify feature separability.
  * Utilized **t-SNE** for 2D dimensionality reduction to visually prove the inherent overlap (Bayes error) between challenging classes (Class 2 vs. Class 5).

##  How to Run

1. Clone this repository.
2. Ensure you have the datasets placed in the `data/` folder.
3. Open the `.ipynb` file using Jupyter Notebook or Google Colab.
4. Run the cells sequentially. The notebook contains pre-rendered visualizations (3D plots, decision boundaries, t-SNE scatter plots) for immediate review.

##  Technologies Used
* **Python** (NumPy, Pandas) for vectorized data manipulation.
* **Scikit-Learn** for SVM implementation, t-SNE, and baseline comparisons.
* **Matplotlib / Seaborn** for 2D and 3D data visualization.
