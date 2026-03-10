import matplotlib.pyplot as plt
from data.dataset import load_dataset
from model.quantum_model import quantum_model
from training.train import train
from evaluation.metrics import evaluate


X_train, X_test, y_train, y_test = load_dataset()

n_params = 2

params_opt, loss_history = train(
    X_train,
    y_train,
    quantum_model,
    n_params
)

accuracy = evaluate(X_test, y_test, params_opt, quantum_model)

print("Test Accuracy:", accuracy)

plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()