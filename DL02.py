import numpy as np
import matplotlib.pyplot as plt

# 데이터: f(x)=4x(1-x)
X_tr = np.arange(0.0, 1.01, 0.1).reshape(-1,1)
y_tr = 4*X_tr*(1-X_tr)
X_val = np.arange(0.05, 1.00, 0.10).reshape(-1,1)
y_val = 4*X_val*(1-X_val)

# 모델
np.random.seed(42)
hidden = 4
lr = 0.7

epochs = 500000

W1 = np.random.randn(1, hidden)*0.1
b1 = np.zeros((1, hidden))
W2 = np.random.randn(hidden, 1)*0.1
b2 = np.zeros((1, 1))

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1/(1+np.exp(-z))
def d_sigmoid(a): return a*(1-a)
def mse(y, yhat): return np.mean((y-yhat)**2)

train_loss, val_loss = [], []
best_val = np.inf
best_epoch = 0
best_params = None

for ep in range(epochs):
    a1_tr = sigmoid(X_tr @ W1 + b1)
    yhat_tr = sigmoid(a1_tr @ W2 + b2)

    a1_val = sigmoid(X_val @ W1 + b1)
    yhat_val = sigmoid(a1_val @ W2 + b2)

    Ltr = mse(y_tr, yhat_tr)
    Lval = mse(y_val, yhat_val)
    train_loss.append(Ltr)
    val_loss.append(Lval)

    if Lval < best_val:
        best_val = Lval
        best_epoch = ep
        best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

    dE = (yhat_tr - y_tr) * (2 / y_tr.shape[0])
    dZ2 = dE * d_sigmoid(yhat_tr)
    dW2 = a1_tr.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * d_sigmoid(a1_tr)
    dW1 = X_tr.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 -= lr*dW2; b2 -= lr*db2
    W1 -= lr*dW1; b1 -= lr*db1

print(f"Early stopping position (val min): epoch={best_epoch}, val_MSE={best_val:.6f}")

# 스탑 포지션
plt.figure(figsize=(7,4))
plt.plot(train_loss, label="Train MSE")
plt.plot(val_loss,   label="Val MSE")
plt.axvline(best_epoch, linestyle="--", alpha=0.9, color="tab:red")
plt.text(best_epoch, max(val_loss)*0.95, f" stop @ {best_epoch}", rotation=90,
         va='top', ha='right', color="tab:red")
plt.title("Train vs Val Loss (f(x)=4x(1-x))")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout(); plt.show()

# best snapshot: 예측선 vs 실제값
W1b, b1b, W2b, b2b = best_params
xx = np.linspace(0,1,400).reshape(-1,1)
yy_nn = sigmoid(sigmoid(xx @ W1b + b1b) @ W2b + b2b)

plt.figure(figsize=(6,4))
plt.scatter(X_tr, y_tr, color='red', s=40, label='Actual Data')
plt.plot(xx, yy_nn, color='blue', label='Predicted (best)')
plt.xlabel('Input (x)'); plt.ylabel('f(x)')
plt.title('Neural Network Approximation of f(x) = 4x(1-x)')
plt.ylim(-0.05,1.05); plt.xlim(0,1); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# 훈련하지 않은 포인트들
X_test = np.array([[0.05],[0.125],[0.275],[0.55],[0.725],[0.95]])
y_test_true = 4*X_test*(1-X_test)

# 얼리스탑 포인트
def predict(Xin, W1, b1, W2, b2):
    a1 = sigmoid(Xin @ W1 + b1)
    return sigmoid(a1 @ W2 + b2)

y_test_pred = predict(X_test, W1b, b1b, W2b, b2b)
test_mse = np.mean((y_test_true - y_test_pred)**2)
print("\n[Generalization on unseen points]")
for x, yt, yp in zip(X_test.ravel(), y_test_true.ravel(), y_test_pred.ravel()):
    print(f"x={x:>6.3f}  f(x)={yt:>7.4f}  NN={yp:>7.4f}")
print("Test MSE:", float(test_mse))

# 시각화 : 트레이닝 데이터(red), NN 곡선(blue), 훈련안시킨(안보이는) 데이터(green)
plt.figure(figsize=(6,4))
plt.scatter(X_tr, y_tr, color='red',  s=40, label='Train data')
plt.scatter(X_test, y_test_true, color='green', s=55, marker='D', label='Unseen targets')
plt.plot(xx, yy_nn, color='blue', label='Predicted (best)')
plt.xlabel('Input (x)'); plt.ylabel('f(x)')
plt.title('Generalization on Unseen Points')
plt.ylim(-0.05,1.05); plt.xlim(0,1); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()