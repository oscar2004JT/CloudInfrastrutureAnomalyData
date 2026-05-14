import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)

# ---------- CARGA DE MODELO Y DATOS ----------
def cargar_modelo_y_datos(modelo_path, test_csv):
    modelo = joblib.load(modelo_path)
    df_test = pd.read_csv(test_csv)
    return modelo, df_test

# ---------- PARA SKLEARN ----------
def preparar_X_y_test(modelo, df_test, target_col='Anomaly status'):
    cols_modelo = modelo.feature_names_in_
    X_test = df_test[cols_modelo]
    y_test = df_test[target_col].astype(int)
    return X_test, y_test

def calcular_probabilidades(modelo, X_test):
    return modelo.predict_proba(X_test)[:, 1]

# ---------- PARA DNN/KERAS ----------
def preparar_X_y_test_dnn(df_test, target_col='Anomaly status'):
    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].astype(int).values
    return X_test, y_test

def calcular_probabilidades_dnn(modelo, X_test):
    probs = modelo.predict(X_test)
    if probs.ndim > 1 and probs.shape[1] > 1:
        return probs[:, 1]
    return probs.flatten()

# ---------- BÚSQUEDA DE UMBRAL ÓPTIMO ----------
def buscar_umbral_optimo(probs, y_test, target_recall_0=0.90, target_recall_1=0.85):
    thresholds = np.linspace(0.0, 1.0, 1001)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        r1 = recall_score(y_test, preds, pos_label=1)
        r0 = recall_score(y_test, preds, pos_label=0)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        results.append({
            'threshold': t,
            'accuracy': acc,
            'precision': prec,
            'recall_0': r0,
            'recall_1': r1,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        })

    df_res = pd.DataFrame(results)
    mask_both = (df_res['recall_0'] >= target_recall_0) & (df_res['recall_1'] >= target_recall_1)

    if mask_both.any():
        best_ok = df_res[mask_both].sort_values('accuracy', ascending=False).iloc[0]
    else:
        df_res['dist_sum'] = (np.abs(df_res['recall_0'] - target_recall_0)
                             + np.abs(df_res['recall_1'] - target_recall_1))
        best_ok = df_res.sort_values('dist_sum').iloc[0]

    return df_res, best_ok['threshold'], best_ok

# ---------- GUARDAR MÉTRICAS Y PREDICCIONES ----------
def guardar_metricas_y_predicciones(df_res, probs, y_test, t_use,
                                    metrica_csv='metricas_por_umbral.csv',
                                    preds_csv='Predicciones_Test_Ajustadas_umbral.csv'):
    df_res.to_csv(metrica_csv, index=False)
    preds_final = (probs >= t_use).astype(int)
    df_out = pd.DataFrame({
        'Probabilidad_Anomalia': probs,
        'Prediccion': preds_final,
        'Real': y_test
    })
    df_out.to_csv(preds_csv, index=False)
    return preds_final, df_out

# ---------- RESUMEN DEL UMBRAL ----------
def resumen_umbral(best_row, y_test, preds_final):
    print("\n====== RESUMEN DEL UMBRAL ÓPTIMO ======")
    print(f"Umbral seleccionado: {best_row['threshold']:.3f}")
    print(f"Accuracy: {best_row['accuracy']:.4f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall clase 0 (No Anómala): {best_row['recall_0']:.4f}")
    print(f"Recall clase 1 (Anómala): {best_row['recall_1']:.4f}")
    print("----------------------------------------")
    print(f"Total de muestras: {len(y_test)}")
    print(f"Anomalías predichas: {(preds_final == 1).sum()}")
    print(f"No anomalías predichas: {(preds_final == 0).sum()}")
    print("----------------------------------------")
    print("Reporte de clasificación:")
    print(classification_report(y_test, preds_final, target_names=['No Anómala', 'Anómala'], zero_division=0))
    print("----------------------------------------")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, preds_final))
    print("========================================\n")

# ---------- GRÁFICOS UNIFICADOS ----------
def graficar_todo(modelo=None, X_test=None, y_test=None, probs=None, umbral=0.5, tipo='sklearn'):
    """
    Grafica todos los reportes y métricas.
    tipo: 'sklearn' o 'dnn'
    """
    if tipo == 'sklearn':
        # Predicciones y probabilidades
        y_pred = modelo.predict(X_test)
        y_score = modelo.predict_proba(X_test)[:,1]
    else:  # DNN
        if probs is None:
            probs = calcular_probabilidades_dnn(modelo, X_test)
        y_pred = (probs >= umbral).astype(int)
        y_score = probs

    # --- Matriz de Confusión ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.colorbar()
    classes = ['No Anómala', 'Anómala']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Etiqueta real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.show()

    # --- Barras Real vs Predicho ---
    plt.figure(figsize=(6,5))
    real_counts = np.bincount(y_test)
    pred_counts = np.bincount(y_pred)
    plt.bar([0,1], real_counts, alpha=0.6, label='Reales')
    plt.bar([0,1], pred_counts, alpha=0.6, label='Predichos')
    plt.xticks([0,1], classes)
    plt.ylabel("Cantidad")
    plt.title("Conteo de clases reales vs predichas")
    plt.legend()
    plt.show()

    # --- Curva ROC ---
    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title("Curva ROC")
    plt.show()

    # --- Curva Precision-Recall ---
    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title("Curva Precision–Recall")
    plt.show()

# ---------- MAIN SKLEARN ----------
def main(modelo_path, test_csv, target_recall_0=0.90, target_recall_1=0.85):
    modelo, df_test = cargar_modelo_y_datos(modelo_path, test_csv)
    X_test, y_test = preparar_X_y_test(modelo, df_test)
    probs = calcular_probabilidades(modelo, X_test)
    df_res, t_use, best_row = buscar_umbral_optimo(probs, y_test, target_recall_0, target_recall_1)
    preds_final, df_out = guardar_metricas_y_predicciones(df_res, probs, y_test, t_use)
    resumen_umbral(best_row, y_test, preds_final)
    graficar_todo(modelo=modelo, X_test=X_test, y_test=y_test, tipo='sklearn')
    return df_res, preds_final, t_use, df_out

# ---------- MAIN DNN ----------
def main_dnn(modelo_path, test_csv, target_recall_0=0.90, target_recall_1=0.85):
    modelo, df_test = cargar_modelo_y_datos(modelo_path, test_csv)
    X_test, y_test = preparar_X_y_test_dnn(df_test)
    probs = calcular_probabilidades_dnn(modelo, X_test)
    df_res, t_use, best_row = buscar_umbral_optimo(probs, y_test, target_recall_0, target_recall_1)
    preds_final, df_out = guardar_metricas_y_predicciones(df_res, probs, y_test, t_use)
    resumen_umbral(best_row, y_test, preds_final)
    graficar_todo(y_test=y_test, probs=probs, umbral=t_use, tipo='dnn')
    return df_res, preds_final, t_use, df_out
