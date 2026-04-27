Logistic Regression From Scratch
___________________
A clean, minimal, and fully‑explained implementation of Logistic Regression from scratch using only NumPy.

This project demonstrates how the core components of logistic regression work internally, without relying on machine‑learning libraries such as scikit‑learn.

🚀 Features
Manual implementation of:
Sigmoid activation
Binary Cross‑Entropy (BCE) Loss
Gradient computation
Weight & bias update (Gradient Descent optimizer)
Fully vectorized NumPy operations
Customizable learning rate and number of epochs
Training loop with loss tracking
Prediction using a probability threshold (default = 0.5)
Clean, readable class‑based structure
📌 Project Structure
text
.
├── logistic_regression.py     # Model implementation
├── main.py                    # Example usage / training script
└── README.md                  # Project documentation
🧠 How It Works
1. Forward Pass
Computes:

Linear combination:z = X · w + b
Sigmoid activation:y_hat = 1 / (1 + exp(-z))
2. Loss Function
Binary Cross‑Entropy:

Loss = - ( y*log(y_hat) + (1 - y)*log(1 - y_hat) )
3. Gradient Calculation
grad_w = X.T · (y_hat - y) / N
grad_b = sum(y_hat - y) / N
4. Weight Update (Optimizer)
(Stochastic Gradient Descent)

w = w - lr * grad_w
b = b - lr * grad_b

🧪 Usage Example
python
import numpy as np
from logistic_regression import LogisticRegression

# Sample dataset
X = np.array([
    [1, 2, 3],
    [2, 1, 0],
    [3, 1, 2],
    [1, 0, 1],
    [2, 3, 2]
])
y = np.array([1, 0, 1, 0, 1])

# Create model
model = LogisticRegression()

# Train
model.train(lr=0.1)

# Predict
preds = model.predict(X)
print("Predictions:", preds)
📊 Example Output
text
Epoch 100 | Loss: 0.532
Epoch 200 | Loss: 0.431
Epoch 300 | Loss: 0.366
...
Predictions: [1 0 1 0 1]
🧩 TODO / Future Improvements
Add regularization (L2 / L1)
Add support for batch gradient descent
Implement different optimizers (Momentum, Adam)
Plot loss curve automatically
Add dataset loading utilities
❤️ Contributing
Pull requests and suggestions are welcome!

Feel free to open an issue if you find a bug or want to request a feature.

📄 License
This project is released under the MIT License.

You are free to use, modify, and distribute it.
