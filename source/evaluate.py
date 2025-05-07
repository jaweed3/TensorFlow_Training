# src/evaluate.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
import numpy as np
import json
import sys

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

model = tf.keras.models.load_model("saved_model/MNIST.keras")
preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)

acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {acc:.4f}")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

if acc < 0.90:
    print("❌ Accuracy below threshold. Failing CI.")
    sys.exit(1)
