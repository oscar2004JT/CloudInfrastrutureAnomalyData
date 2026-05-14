import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE

# -------------------- Preprocesamiento --------------------
def convertir_timestamp(df, col='timestamp'):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def extraer_fecha_hora(df, col='timestamp'):
    if col in df.columns:
        df['day'] = df[col].dt.day
        df['month'] = df[col].dt.month
        df['hour'] = df[col].dt.hour
    return df

def eliminar_columnas(df, cols=['vm_id', 'timestamp']):
    return df.drop(columns=[c for c in cols if c in df.columns], errors='ignore')

def cargar_y_limpieza(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df = convertir_timestamp(df, 'timestamp')
    df = extraer_fecha_hora(df, 'timestamp')
    df = eliminar_columnas(df)
    return df

def codificar_categoricas(df):
    cat_cols = ['task_type', 'task_priority', 'task_status']
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = df.dropna(subset=cat_cols)
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def crear_variables_derivadas(df):
    df['io_vs_network_ratio'] = df.get('task_type_io', 0) / (df.get('task_type_network', 0) + 1e-6)
    df['priority_index'] = (df.get('task_priority_low', 0) * 1) + (df.get('task_priority_medium', 0) * 2)
    df['status_ratio'] = df.get('task_status_waiting', 0) / (df.get('task_status_running', 0) + 1e-6)
    df['efficiency_per_cpu'] = df.get('energy_efficiency', 0) / (df.get('cpu_usage', 0) + 1e-6)
    df['traffic_efficiency_ratio'] = df.get('network_traffic', 0) / (df.get('energy_efficiency', 0) + 1e-6)
    df['cpu_memory_balance'] = df.get('cpu_usage', 0) - df.get('memory_usage', 0)
    df['waiting_pressure_index'] = (df.get('task_status_waiting', 0) + df.get('task_priority_low', 0)) / (df.get('task_status_running', 0) + 1)
    df['io_energy_ratio'] = df.get('task_type_io', 0) / (df.get('energy_efficiency', 0) + 1e-6)
    return df

def rellenar_nulos(df):
    for col in df.select_dtypes(include=['uint8']).columns:
        df[col] = df[col].astype('int64')
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df.dropna()
    assert df.isnull().sum().sum() == 0, "Quedan valores NaN en el DataFrame después de limpiar"
    return df

def seleccionar_columnas(df):
    cols_final = [
        'cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption',
        'num_executed_instructions', 'execution_time', 'energy_efficiency',
        'task_type_io', 'task_type_network', 'task_priority_low', 'task_priority_medium',
        'task_status_running', 'task_status_waiting',
        'io_vs_network_ratio', 'priority_index', 'status_ratio', 'efficiency_per_cpu',
        'traffic_efficiency_ratio', 'cpu_memory_balance', 'waiting_pressure_index', 'io_energy_ratio',
        'day', 'month', 'hour',
        'Anomaly status'
    ]
    return df[[c for c in cols_final if c in df.columns]]

# -------------------- División y escalado --------------------
def dividir_X_y(df, target_col='Anomaly status'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def escalar_variables(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled

# -------------------- Balanceo con BorderlineSMOTE --------------------
def balancear_dataset(X, y):
    smote = BorderlineSMOTE(kind='borderline-1', random_state=42)
    return smote.fit_resample(X, y)

# -------------------- Guardado de datasets --------------------
def guardar_datasets_balanceados(X_train_res, y_train_res, X_test_res, y_test_res):
    pd.concat([X_train_res, y_train_res], axis=1).to_csv(
        "Cloud_Anomaly_Dataset_Train_Balanceado_E.csv", index=False
    )
    pd.concat([X_test_res, y_test_res], axis=1).to_csv(
        "Cloud_Anomaly_Dataset_Test_Balanceado_E.csv", index=False
    )
    print("\nArchivos balanceados generados correctamente:")
    print(" - Cloud_Anomaly_Dataset_Train_Balanceado_E.csv")
    print(" - Cloud_Anomaly_Dataset_Test_Balanceado_E.csv")

# -------------------- Función principal --------------------
def main():
    df = cargar_y_limpieza("Cloud_Anomaly_Dataset.csv")
    df = codificar_categoricas(df)
    df = crear_variables_derivadas(df)
    df = rellenar_nulos(df)
    df = seleccionar_columnas(df)

    X, y = dividir_X_y(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Escalado
    X_train_scaled, X_test_scaled = escalar_variables(X_train, X_test)

    print(f"Distribución de la variable objetivo en el conjunto de entrenamiento (sin balancear):\n{y_train.value_counts()}")

    # Balanceo
    X_train_res, y_train_res = balancear_dataset(X_train_scaled, y_train)
    print(f"Distribución de la variable objetivo después de balancear el conjunto de entrenamiento:\n{y_train_res.value_counts()}")

    X_test_res, y_test_res = balancear_dataset(X_test_scaled, y_test)
    print(f"Distribución de la variable objetivo después de balancear el conjunto de prueba:\n{y_test_res.value_counts()}")

    # Guardar datasets balanceados
    guardar_datasets_balanceados(X_train_res, y_train_res, X_test_res, y_test_res)

    print(f"Dimensiones de los conjuntos balanceados:\nX_train_res: {X_train_res.shape}\nX_test_res: {X_test_res.shape}")

if __name__ == "__main__":
    main()
