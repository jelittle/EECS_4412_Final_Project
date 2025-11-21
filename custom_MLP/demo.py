import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as sklearn_mlp
from custom_mlp import mlpClassifier as mlp
CONFIGS = {
    "data_path": "../project_data/t4sa_data.csv",


}


def build_mlp():
    model=mlp(layers=[300,100,50], epochs=300,activation="relu", learning_rate=0.001, batch_size=128, verbose=True)
    sk_mlp=sklearn_mlp(hidden_layer_sizes=(300,100,50), max_iter=300, activation="relu", learning_rate_init=0.01, batch_size=128,verbose=True)
    return model, sk_mlp

def runner(configs):
    model, sk_mlp=build_mlp()
    data_path = configs.get("data_path")
    data = pd.read_csv(data_path)
    # Parse the "embeddings" column from string to list
    # data['embeddings'] = data['embeddings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    print("parsing embeddings...")
    if isinstance(data['embeddings'].iloc[0], str):
        # Remove brackets and split
        data['embeddings'] = data['embeddings'].str.strip('[]').str.split(',').apply(
            lambda x: np.array([float(i) for i in x])
        )
    
    X,y= data['embeddings'], data['class']
    # Encode unique y values to unique integers(needed for mlp)
    y_unique = {label: idx for idx, label in enumerate(sorted(y.unique()))}
    y = y.map(y_unique)
    X = np.stack(X.values)
    # First split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    print(len(X_train))
    model.fit(X_train, y_train, X_val, y_val)
    # sk_mlp.fit(X_train, y_train)
    acc_sk= sk_mlp.score(X_test, y_test)
    print(f"baseline Accuracy: {acc_sk}")
    acc= model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc}")





if __name__ == "__main__":

    runner(CONFIGS)

