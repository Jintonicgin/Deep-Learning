import numpy as np
import matplotlib.pyplot as plt

# 입력 데이터
X = np.array([
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [1., 1.]
])
y = np.array([[0.], [1.], [1.], [0.]])

lr = 0.02
epochs = 100000

# 가중치 초기화 
W1 = np.array([[0.40, -0.30],
               [-0.35, 0.45]], dtype=float)
b1 = np.array([0.30, 0.30], dtype=float)  # 첫 번째 은닉층 바이어스

W2 = np.array([[0.50], [-0.60]], dtype=float)
b2 = np.array([0.40], dtype=float)        # 출력층 바이어스

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def d_sigmoid_from_output(a): return a * (1.0 - a)
def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)

loss_hist = []

for _ in range(epochs):
    # 순전파
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)

    loss_hist.append(mse(y, y_hat))

    # 역전파
    dE_dyhat = (y_hat - y)
    dE_dz2 = dE_dyhat * d_sigmoid_from_output(y_hat)
    dE_dW2 = a1.T @ dE_dz2
    dE_db2 = np.sum(dE_dz2, axis=0)

    dE_da1 = dE_dz2 @ W2.T
    dE_dz1 = dE_da1 * d_sigmoid_from_output(a1)
    dE_dW1 = X.T @ dE_dz1
    dE_db1 = np.sum(dE_dz1, axis=0)

    # 가중치 업데이트
    W2 -= lr * dE_dW2
    b2 -= lr * dE_db2
    W1 -= lr * dE_dW1
    b1 -= lr * dE_db1

# 최종 출력
z1 = X @ W1 + b1
a1 = sigmoid(z1)
z2 = a1 @ W2 + b2
y_hat = sigmoid(z2)

pred = (y_hat >= 0.5).astype(int)
acc = float(np.mean(pred == y))

print("[100000번 학습 결과]")
print("Final outputs (prob.):", y_hat.ravel().tolist())
print("Predictions:", pred.ravel().tolist())
print("Accuracy:", acc)

print("\nFinal Weights (Input -> Hidden):")
print(W1.tolist())
print("Final Biases (Hidden):", b1.tolist())

print("\nFinal Weights (Hidden -> Output):")
print(W2.ravel().tolist())
print("Final Bias (Output):", b2.tolist())

plt.figure(figsize=(6, 4))
plt.plot(loss_hist)
plt.title("Error reduction (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.tight_layout()
plt.show()