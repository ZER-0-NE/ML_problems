### 📋 What is a vector?

* A vector is a list (tuple) of numbers, often called **scalar values**:

  Each $v_i$ is a real number, and $n$ is the **dimensionality** of the vector ([machinelearningmastery.com][1]).

* In code (like NumPy), that's exactly what it is: a 1D array:

  ```python
  import numpy as np
  v = np.array([1.0, 2.5, -0.3])  # vector of length 3 (3-dimensional)
  ```

  A vector of length n = an n-dimensional vector .

---

### 📏 Why call it a vector?

Because it behaves like a mathematical object in **n-dimensional space**:

* Each number is a **coordinate** along one axis.
* You can do math with it—like add two vectors or scale them by a number.
* It's exactly like a point or arrow in an n-dimensional coordinate system ([stackoverflow.com][2], [math.stackexchange.com][3], [en.wikipedia.org][4]).

---

### 🔢 Comparison: array vs. vector vs. tensor

* **1D array** = vector.
* **2D array** = matrix.
* **ND array (N ≥ 3)** = tensor ([numpy.org][5], [neptune.ai][6]).
* Machine learning uses vectors to represent data, features, embeddings, weights, etc. ([shelf.io][7]).

---
