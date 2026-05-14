import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, confusion_matrix,
    classification_report, mean_squared_error, mean_absolute_error, r2_score
)

def cargar_modelo_y_datos(modelo_path, test_csv):
    modelo = joblib.load(modelo_path)
    df_test = pd.read_csv(test_csv)
    return modelo, df_test

def preparar_X_y_test(modelo, df_test, target_col='Anomaly status'):
    cols_modelo = modelo.feature_names_in_
    X_test = df_test[cols_modelo]
    y_test = df_test[target_col].astype(int)
    return X_test, y_test

def calcular_probabilidades(modelo, X_test):
    return modelo.predict_proba(X_test)[:, 1]

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
        mse = mean_squared_error(y_test, probs)
        mae = mean_absolute_error(y_test, probs)
        r2 = r2_score(y_test, probs)
        results.append({
            'threshold': t,
            'accuracy': acc,
            'precision': prec,
            'recall_0': r0,
            'recall_1': r1,
            'mse': mse,
            'mae': mae,
            'r2': r2,
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
    t_use = best_ok['threshold']
    return df_res, t_use, best_ok

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

def resumen_umbral(best_row, y_test, preds_final):
    print("\n====== RESUMEN DEL UMBRAL ÓPTIMO ======")
    print(f"Umbral seleccionado: {best_row['threshold']:.3f}")
    print(f"Accuracy: {best_row['accuracy']:.4f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall clase 0 (No Anómala): {best_row['recall_0']:.4f}")
    print(f"Recall clase 1 (Anómala): {best_row['recall_1']:.4f}")
    print(f"MSE: {best_row['mse']:.6f}")
    print(f"MAE: {best_row['mae']:.6f}")
    print(f"R²: {best_row['r2']:.6f}")
    print("----------------------------------------")
    print(f"Total de muestras: {len(y_test)}")
    print(f"Anomalías predichas: {(preds_final == 1).sum()}")
    print(f"No anomalías predichas: {(preds_final == 0).sum()}")
    print("----------------------------------------")
    print("Reporte de clasificación:")
    print(classification_report(y_test, preds_final, target_names=['No Anómala', 'Anómala']))
    print("----------------------------------------")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, preds_final))
    print("========================================\n")

def main(modelo_path, test_csv, target_recall_0=0.90, target_recall_1=0.85):
    modelo, df_test = cargar_modelo_y_datos(modelo_path, test_csv)
    X_test, y_test = preparar_X_y_test(modelo, df_test)
    probs = calcular_probabilidades(modelo, X_test)
    df_res, t_use, best_row = buscar_umbral_optimo(probs, y_test, target_recall_0, target_recall_1)
    preds_final, df_out = guardar_metricas_y_predicciones(df_res, probs, y_test, t_use)
    resumen_umbral(best_row, y_test, preds_final)
    return df_res, preds_final, t_use, df_out
