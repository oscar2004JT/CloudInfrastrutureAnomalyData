# logistic_regression_funciones.py
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Función para cargar datasets
def cargar_datasets(train_path, test_path, target_col='Anomaly status'):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    return X_train, X_test, y_train, y_test

# Función para entrenar Logistic Regression para clasificación binaria
def entrenar_logistic_regression_binaria(X_train, y_train, param_grid=None, cv=2, scoring='f1'):
    
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],  # buenos para binaria
            'class_weight': ['balanced'],
            'max_iter': [500, 1000]
        }
    
    lr = LogisticRegression()

    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=2,
        refit=True
    )
    
    print("Buscando los mejores hiperparámetros para Regresión Logística Binaria...")
    grid_search.fit(X_train, y_train)
    
    print("\nMejores parámetros:", grid_search.best_params_)
    print(f"Mejor F1 en validación cruzada: {grid_search.best_score_:.4f}")
    
    pd.DataFrame(grid_search.cv_results_).to_csv('Resultados_GridSearch_LogReg_Binaria.csv', index=False)
    print("Resultados completos guardados como 'Resultados_GridSearch_LogReg_Binaria.csv'")
    
    return grid_search.best_estimator_

# Función para evaluar modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    
    print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, zero_division=0))

# Importancia de variables para LR binaria = coeficientes absolutos
def importancia_variables(modelo, X_train, plot=True):
    coef = modelo.coef_[0]  # solo una fila porque es binaria
    importancia = pd.DataFrame({
        'Variable': X_train.columns,
        'Importancia': abs(coef)
    }).sort_values(by='Importancia', ascending=False)
    
    print("\nTop variables del modelo:\n")
    print(importancia.to_string(index=False))
    
    importancia.to_csv('Importancia_Variables_LogReg_Binaria.csv', index=False)
    print("Importancias guardadas como 'Importancia_Variables_LogReg_Binaria.csv'")
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.barh(importancia['Variable'][::-1], importancia['Importancia'][::-1])
        plt.title('Importancia de Variables (Regresión Logística Binaria)')
        plt.xlabel('|Coeficiente|')
        plt.ylabel('Variable')
        plt.tight_layout()
        plt.show()
    
    return importancia

# Guardar modelo
def guardar_modelo(modelo, nombre_archivo='modelo_logistic_regression_binaria.pkl'):
    joblib.dump(modelo, nombre_archivo)
    print(f"Modelo guardado como '{nombre_archivo}'")

# Función para graficar lo que el modelo predijo vs lo real
def graficar_real_vs_predicho(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.6, label='Real')
    plt.scatter(range(len(y_test)), y_pred, color='red', alpha=0.6, label='Predicho', marker='x')
    
    plt.title('Valores Reales vs Predichos')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Clase')
    plt.yticks([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.show()

# Función principal
def main():
    X_train, X_test, y_train, y_test = cargar_datasets(
        'Cloud_Anomaly_Dataset_Train_Balanceado_E.csv',
        'Cloud_Anomaly_Dataset_Test_E.csv'
    )
    
    best_lr = entrenar_logistic_regression_binaria(X_train, y_train)
    evaluar_modelo(best_lr, X_test, y_test)
    importancia_variables(best_lr, X_train)
    guardar_modelo(best_lr)

if __name__ == "__main__":
    main()
