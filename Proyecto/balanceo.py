import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE

# ================================
# 1. Conversión y limpieza inicial
# ================================
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

def eliminar_columnas(df, cols=['vm_id', 'timestamp', 'cpu_per_instruction',
                                'memory_per_task', 'power_per_time', 'traffic_per_cpu']):
    df = df.drop(columns=[c for c in cols if c in df.columns], errors='ignore')
    return df

def cargar_y_limpieza(ruta_csv):
    df = pd.read_csv(ruta_csv)
    df = convertir_timestamp(df, 'timestamp')
    df = extraer_fecha_hora(df, 'timestamp')
    df = eliminar_columnas(df)
    return df

# ================================
# 2. Codificación de categóricas
# ================================
def codificar_categoricas(df):
    cat_cols = ['task_type', 'task_priority', 'task_status']
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

# ================================
# 3. Crear variables derivadas
# ================================
def crear_variables_derivadas(df):
    df['io_vs_network_ratio'] = df['task_type_io'] / (df['task_type_network'] + 1e-6)
    df['priority_index'] = (df['task_priority_low'] * 1) + (df['task_priority_medium'] * 2)
    df['status_ratio'] = df['task_status_waiting'] / (df['task_status_running'] + 1e-6)
    df['efficiency_per_cpu'] = df['energy_efficiency'] / (df['cpu_usage'] + 1e-6)
    df['traffic_efficiency_ratio'] = df['network_traffic'] / (df['energy_efficiency'] + 1e-6)
    df['cpu_memory_balance'] = df['cpu_usage'] - df['memory_usage']
    df['waiting_pressure_index'] = (df['task_status_waiting'] + df['task_priority_low']) / (df['task_status_running'] + 1)
    df['io_energy_ratio'] = df['task_type_io'] / (df['energy_efficiency'] + 1e-6)
    return df

# ================================
# 4. Imputación de nulos
# ================================
def rellenar_nulos(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

# ================================
# 5. Selección final
# ================================
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
    df = df[[c for c in cols_final if c in df.columns]]
    return df

# ================================
# 6. Dividir X y y
# ================================
def dividir_X_y(df, target_col='Anomaly status'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# ================================
# 7. División Train/Test
# ================================
def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# ================================
# 8. Balanceo con SMOTE
# ================================
def balancear_dataset(X, y):
    smote = BorderlineSMOTE(kind='borderline-1', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

# ================================
# 9. Guardar datasets procesados
# ================================
def guardar_datasets(df, X_train_res, y_train_res, X_test, y_test, X_test_res, y_test_res):
    df.to_csv("Cloud_Anomaly_Dataset_Limpio.csv", index=False)
    pd.concat([X_train_res, y_train_res], axis=1).to_csv("Cloud_Anomaly_Dataset_Train_Balanceado.csv", index=False)
    
    # Test SIN balancear
    pd.concat([X_test, y_test], axis=1).to_csv("Cloud_Anomaly_Dataset_Test.csv", index=False)

    # Test balanceado
    pd.concat([X_test_res, y_test_res], axis=1).to_csv("Cloud_Anomaly_Dataset_Test_Balanceado.csv", index=False)

    print("Archivos generados:")
    print(" - Cloud_Anomaly_Dataset_Limpio.csv")
    print(" - Cloud_Anomaly_Dataset_Train_Balanceado.csv")
    print(" - Cloud_Anomaly_Dataset_Test.csv     (NO balanceado)")
    print(" - Cloud_Anomaly_Dataset_Test_Balanceado.csv")
