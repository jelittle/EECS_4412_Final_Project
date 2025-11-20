import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from custom_mlp import mlp

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

def train_custom_mlp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = mlp(layers=[2,4,1], epochs=300, learning_rate=0.01, batch_size=10)

    model.fit(X_train, y_train, X_test, y_test)

    # print(train_data)
    # print(len(train_data[0]))
    # model.fit(training_data=train_data, epochs=300, batch_size=10, lr=0.01, test_data=test_data)
    # model.fit(training_data=train_data,test_data=test_data)

    
    return model

if __name__ == "__main__":
    X, y = generate_xor_dataset()
    # baseline_model = train_baseline_mlp(X, y)
    custom_model = train_custom_mlp(X, y)
