# ======================================================
#   IMPORTANCIA DE VARIABLES PARA MLP (FUNCTION + PLOT)
# ======================================================

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np


def obtener_importancia_variables(modelo, X_test, y_test, n_repeats=10):
    """
    Calcula la importancia de variables usando Permutation Importance
    para un modelo MLP ya entrenado.

    Retorna:
        - importances: array con importancia promedio
        - indices: orden de importancia descendente
        - feature_names: nombres de las columnas
    """
    result = permutation_importance(
        modelo,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]
    feature_names = X_test.columns

    return importances, indices, feature_names



def graficar_importancia_variables(importances, indices, feature_names, titulo="Importancia de Variables - MLP"):
    """
    Grafica la importancia de variables obtenida con permutation importance.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.title(titulo)
    plt.xlabel("Variables")
    plt.ylabel("Importancia promedio")
    plt.tight_layout()
    plt.show()
