import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from custom_mlp import mlp
from mlp_archives.mlp_v1 import mlp as mlp_v1
from mlp_archives.mlp_v2 import mlp as mlp_v2
import time
import psutil
def generate_xor_dataset(n_samples=1000, noise_std=0.1, random_seed=42):
    X_original = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_original = np.array([0, 1, 1, 0])
    np.random.seed(random_seed)
    X, y = [], []
    for _ in range(n_samples):
        idx = np.random.randint(0, 4)
        noise = np.random.normal(0, noise_std, 2)
        X.append(X_original[idx] + noise)
        y.append(y_original[idx])
    X = np.array(X)
    y = np.array(y)
    print(f"Expanded dataset size: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y

def train_baseline_mlp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp_clf.fit(X_train, y_train)
    y_pred = mlp_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Baseline MLP Accuracy: {accuracy}")
    return mlp_clf
def track_performance(func, *args, **kwargs):
    """
    track performance metrics.
    """
    process = psutil.Process()

    mem_start = process.memory_info().rss / (1024 * 1024)  # in MB
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    mem_end = process.memory_info().rss / (1024 * 1024)  # in MB
    elapsed = end_time - start_time

    mem_used = mem_end - mem_start

    return result, elapsed, mem_used

def train_custom_mlp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    

    
    model = mlp(layers=[4], epochs=300, learning_rate=0.01, batch_size=10, verbose=True)

    # Track performance of model.fit
    _,tim,mem=track_performance(model.fit, X_train, y_train, X_test, y_test)
    acc= accuracy_score(y_test, model.predict(X_test))
    print(f"Custom MLP Accuracy: {acc}, Time: {tim:.2f}s, Memory: {mem:.2f}MB")

    
    return model

def mlp_performance_tests(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    

    models = {
        "mlp_v1": mlp_v1(layers=[4], epochs=300, learning_rate=0.01, batch_size=10),
        "mlp_v2": mlp_v2(hidden_layers=[4], epochs=300, learning_rate=0.01, batch_size=10),
        # Add more models here if needed
    }
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        _,tim,mem=track_performance(model.fit, X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc, time, mem


    for name, acc in results.items():
        print("----------------------------")
        print(f"{name}: Accuracy = {acc}, Time = {tim:.2f}s, Memory = {mem:.2f}MB")
        print("----------------------------")
    return results

if __name__ == "__main__":
    X, y = generate_xor_dataset()
    # baseline_model = train_baseline_mlp(X, y)
    custom_model = train_custom_mlp(X, y)
