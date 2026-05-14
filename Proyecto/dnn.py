import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def cargar_datasets(train_path, test_path, target_col='Anomaly status'):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    return X_train, X_test, y_train, y_test

def crear_modelo(input_dim, n_layers=2, n_neurons=64, lr=0.001, dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    for _ in range(n_layers - 1):
        model.add(Dense(n_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def entrenar_dnn(X_train, y_train):
    clf = KerasClassifier(model=crear_modelo, input_dim=X_train.shape[1], verbose=0)
    param_grid = {
        "model__n_layers": [1, 3],
        "model__n_neurons": [10],
        "model__dropout_rate": [0.2],
        "model__lr": [0.001, 0.0005],
        "epochs": [20],
        "batch_size": [32]
    }
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, scoring="f1", verbose=2, n_jobs=-1, refit=True)
    print("Buscando los mejores hiperparámetros...")
    grid.fit(X_train, y_train)
    pd.DataFrame(grid.cv_results_).to_csv("Resultados_GridSearch_DNN.csv", index=False)
    print("Mejores parámetros encontrados:", grid.best_params_)
    print(f"Mejor F1 en validación cruzada: {grid.best_score_:.4f}")
    return grid.best_estimator_

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = (modelo.predict(X_test) > 0.5).astype(int)
    print("\n--- Resultados del modelo ---")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred, zero_division=0))

def importancia_variables(modelo, X_train, y_train, plot=True):
    r = permutation_importance(modelo, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    importancias = pd.DataFrame({"Variable": X_train.columns, "Importancia": r.importances_mean}).sort_values(by="Importancia", ascending=False)
    importancias.to_csv("Importancia_Variables_DNN.csv", index=False)
    print("\nTop variables según importancia:")
    print(importancias.to_string(index=False))
    if plot:
        plt.figure(figsize=(10, 6))
        plt.barh(importancias["Variable"][::-1], importancias["Importancia"][::-1])
        plt.title("Importancia de variables (DNN)")
        plt.tight_layout()
        plt.show()
    return importancias

def guardar_modelo(modelo, nombre_archivo="modelo_dnn_optimo.pkl"):
    joblib.dump(modelo, nombre_archivo)
    print(f"Modelo guardado como '{nombre_archivo}'")

def main():
    X_train, X_test, y_train, y_test = cargar_datasets(
        "Cloud_Anomaly_Dataset_Train_Balanceado_E.csv",
        "Cloud_Anomaly_Dataset_Test_Balanceado_E.csv"
    )
    best_dnn = entrenar_dnn(X_train, y_train)
    evaluar_modelo(best_dnn, X_test, y_test)
    importancia_variables(best_dnn, X_train, y_train)
    guardar_modelo(best_dnn)

if __name__ == "__main__":
    main()
