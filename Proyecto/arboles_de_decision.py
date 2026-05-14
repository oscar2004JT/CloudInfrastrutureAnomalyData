# decision_tree_funciones.py
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Función para entrenar un Árbol de Decisión con GridSearchCV
def entrenar_arbol_decision(X_train, y_train, param_grid=None, cv=2, scoring='f1'):
    if param_grid is None:
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=2,
        refit=True
    )
    
    print(" Buscando los mejores hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    print("\n Mejores parámetros encontrados:")
    print(grid_search.best_params_)
    print(f"Mejor F1 en validación cruzada: {grid_search.best_score_:.4f}")
    
    pd.DataFrame(grid_search.cv_results_).to_csv('Resultados_GridSearch_DT.csv', index=False)
    print("Resultados completos guardados como 'Resultados_GridSearch_DT.csv'")
    
    return grid_search.best_estimator_

# Evaluar modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    
    print("\n Métricas de evaluación:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, zero_division=0))

# Visualizar importancia de variables
def importancia_variables(modelo, X_train, plot=True):
    importancias = pd.DataFrame({
        'Variable': X_train.columns,
        'Importancia': modelo.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    print("\n Variables más importantes según el Árbol de Decisión:\n")
    print(importancias.to_string(index=False))
    
    importancias.to_csv('Importancia_Variables_DT.csv', index=False)
    print("Importancias guardadas como 'Importancia_Variables_DT.csv'")
    
    if plot:
        plt.figure(figsize=(10,6))
        plt.barh(importancias['Variable'][::-1], importancias['Importancia'][::-1], color='mediumseagreen')
        plt.title('Importancia de variables (Árbol de Decisión)')
        plt.xlabel('Importancia')
        plt.ylabel('Variable')
        plt.tight_layout()
        plt.show()
    
    return importancias

# Guardar modelo
def guardar_modelo(modelo, nombre_archivo='modelo_arbol_decision_optimo.pkl'):
    joblib.dump(modelo, nombre_archivo)
    print(f" Modelo guardado como '{nombre_archivo}'")

# Función principal
def main():
    X_train, X_test, y_train, y_test = cargar_datasets(
        'Cloud_Anomaly_Dataset_Train_Balanceado.csv',
        'Cloud_Anomaly_Dataset_Test.csv'
    )
    
    best_dt = entrenar_arbol_decision(X_train, y_train)
    evaluar_modelo(best_dt, X_test, y_test)
    importancia_variables(best_dt, X_train)
    guardar_modelo(best_dt)

if __name__ == "__main__":
    main()
