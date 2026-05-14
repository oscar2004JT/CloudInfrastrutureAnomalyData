import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

#Función para cargar y preparar el dataset
def cargar_dataset(ruta_csv, columnas_variables, columna_clase="Anomaly status"):
    df = pd.read_csv(ruta_csv)
    
    df.rename(columns={columna_clase: "Anomaly"}, inplace=True)
    
    variables = [v for v in columnas_variables if v in df.columns]
    
    df = df.dropna(subset=variables)
    return df, variables

#Función para separar por clase
def separar_por_clase(df, columna_clase="Anomaly"):
    normal = df[df[columna_clase] == 0]
    anomalo = df[df[columna_clase] == 1]
    return normal, anomalo

#Función para calcular diferencias estadísticas
def calcular_diferencias(normal, anomalo, variables):
    resultados = []
    for var in variables:
        if var == "Anomaly":
            continue
        m0 = normal[var].mean()
        m1 = anomalo[var].mean()
        std_pooled = np.sqrt(((normal[var].std() ** 2) + (anomalo[var].std() ** 2)) / 2)
        cohens_d = (m1 - m0) / std_pooled if std_pooled > 0 else 0
        t_stat, p_val = ttest_ind(normal[var], anomalo[var], equal_var=False)

        resultados.append({
            "Variable": var,
            "Media Clase 0": round(m0, 3),
            "Media Clase 1": round(m1, 3),
            "Diferencia": round(m1 - m0, 3),
            "Cohen's d": round(cohens_d, 3),
            "p-valor": round(p_val, 6)
        })
    return pd.DataFrame(resultados)

#Función para mostrar y guardar resultados
def mostrar_guardar_resultados(df_resultados, ruta_salida="Comparacion_Clases_Extendida.csv"):
    # Ordenar por valor absoluto de Cohen's d
    res_df = df_resultados.sort_values(by="Cohen's d", key=lambda x: abs(x), ascending=False)
    print("\nComparación entre clases (variables seleccionadas y derivadas):\n")
    print(res_df.to_string(index=False))
    res_df.to_csv(ruta_salida, index=False)
    print(f"\nResultados guardados como '{ruta_salida}'")

#Función principal
def main(ruta_csv="Cloud_Anomaly_Dataset.csv", columnas=None):
    if columnas is None:
        columnas = [
            "cpu_usage", "memory_usage", "network_traffic", "power_consumption",
            "num_executed_instructions", "execution_time", "energy_efficiency",
            "task_type_io", "task_type_network",
            "task_priority_low", "task_priority_medium",
            "task_status_running", "task_status_waiting",
            "io_vs_network_ratio", "priority_index", "status_ratio",
            "efficiency_per_cpu", "traffic_efficiency_ratio",
            "cpu_memory_balance", "waiting_pressure_index", "io_energy_ratio"
        ]
    
    df, variables = cargar_dataset(ruta_csv, columnas)
    normal, anomalo = separar_por_clase(df)
    resultados = calcular_diferencias(normal, anomalo, variables)
    mostrar_guardar_resultados(resultados)

if __name__ == "__main__":
    main()
