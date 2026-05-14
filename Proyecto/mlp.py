import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)



def cargar_datos(train_csv, test_csv, target_col="Anomaly status"):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]

    X_test = test.drop(target_col, axis=1)
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test



def entrenar_mlp(X_train, y_train):
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (128, 128)],
        "activation": ["relu"],
        "solver": ["adam"],
        "learning_rate_init": [0.001],
        "max_iter": [200]
    }

    mlp = MLPClassifier(random_state=42)

    grid_mlp = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=2,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    grid_mlp.fit(X_train, y_train)

    best_mlp = grid_mlp.best_estimator_
    best_params = grid_mlp.best_params_

    return best_mlp, best_params



def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    reporte = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, reporte, accuracy, cm




def graficar_matriz_confusion(cm, clases=["Normal", "Anomalía"], titulo="Matriz de Confusión - MLP"):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title(titulo)
    plt.colorbar()

    ticks = np.arange(len(clases))
    plt.xticks(ticks, clases)
    plt.yticks(ticks, clases)

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()




def guardar_modelo(modelo, archivo="modelo_mlp.pkl"):
    with open(archivo, "wb") as file:
        pickle.dump(modelo, file)
    print(f"\nModelo guardado como {archivo}")
