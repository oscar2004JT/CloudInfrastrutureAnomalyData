# CloudInfrastrutureAnomalyData

Proyecto final de Inteligencia Artificial enfocado en la detección de anomalías en infraestructura cloud a partir del dataset `Cloud_Anomaly_Dataset.csv`. El trabajo principal está documentado en el notebook [Proyecto/TrrabajoFinalIA.ipynb](./Proyecto/TrrabajoFinalIA.ipynb) y se apoya en scripts de preprocesamiento, análisis y entrenamiento ubicados en `Proyecto/`.

## Objetivo

Entrenar y comparar modelos supervisados para clasificar registros como normales (`0`) o anómalos (`1`) usando métricas operativas de máquinas virtuales, tráfico, consumo energético y estado de tareas.

Según el notebook, se evaluaron cinco enfoques:

- Árbol de Decisión
- Random Forest
- MLP
- Regresión Logística
- DNN

## Dataset

El conjunto original tiene:

- `277570` registros
- `13` columnas
- variables numéricas, categóricas y temporales
- valores faltantes en varias columnas
- fuerte desbalance de clases: aproximadamente `94 %` normal y `6 %` anómala

Variable objetivo:

- `Anomaly status`

Variables originales destacadas:

- `cpu_usage`
- `memory_usage`
- `network_traffic`
- `power_consumption`
- `num_executed_instructions`
- `execution_time`
- `energy_efficiency`
- `task_type`
- `task_priority`
- `task_status`
- `timestamp`

## Flujo de trabajo del notebook

El pipeline seguido en [Proyecto/TrrabajoFinalIA.ipynb](./Proyecto/TrrabajoFinalIA.ipynb) es:

1. Inspección inicial del dataset.
2. EDA con estadísticas descriptivas, distribución de clases, histogramas y correlación.
3. Comparación entre clases para identificar variables con mayor capacidad discriminativa.
4. Limpieza y transformación de datos.
5. Ingeniería de características.
6. División `train/test`.
7. Balanceo con `BorderlineSMOTE`.
8. Escalado para los modelos sensibles a escala.
9. Entrenamiento y evaluación de modelos.
10. Ajuste de umbral para mejorar el recall por clase.

## Preprocesamiento aplicado

Los scripts [Proyecto/limpieza_y_balanceo.py](./Proyecto/limpieza_y_balanceo.py) y [Proyecto/limpieza_y_balanceoescalado.py](./Proyecto/limpieza_y_balanceoescalado.py) implementan la mayor parte del pipeline:

- conversión de `timestamp` a fecha
- extracción de `day`, `month` y `hour`
- eliminación de columnas irrelevantes como `vm_id` y `timestamp`
- codificación one-hot de:
  - `task_type`
  - `task_priority`
  - `task_status`
- imputación de nulos con la mediana
- selección final de variables
- división estratificada `train/test`
- balanceo con `BorderlineSMOTE`
- escalado con `StandardScaler` para Regresión Logística, MLP y DNN

## Variables derivadas

Con base en el análisis comparativo entre clases, se agregan nuevas variables para capturar relaciones entre recursos y estado de las tareas:

- `io_vs_network_ratio`
- `priority_index`
- `status_ratio`
- `efficiency_per_cpu`
- `traffic_efficiency_ratio`
- `cpu_memory_balance`
- `waiting_pressure_index`
- `io_energy_ratio`

El análisis del notebook destaca especialmente:

- `energy_efficiency`
- `network_traffic`

como las variables con mayor separación entre clases.

## Datasets generados

Durante el preprocesamiento se crean estos archivos:

- `Proyecto/Cloud_Anomaly_Dataset_Limpio.csv`
- `Proyecto/Cloud_Anomaly_Dataset_Train_Balanceado.csv`
- `Proyecto/Cloud_Anomaly_Dataset_Test.csv`
- `Proyecto/Cloud_Anomaly_Dataset_Test_Balanceado.csv`
- `Proyecto/Cloud_Anomaly_Dataset_Train_Balanceado_E.csv`
- `Proyecto/Cloud_Anomaly_Dataset_Test_Balanceado_E.csv`

Los archivos con sufijo `_E` corresponden a versiones escaladas y balanceadas.

## Modelos y scripts

Scripts principales disponibles:

- [Proyecto/arboles_de_decision.py](./Proyecto/arboles_de_decision.py)
- [Proyecto/regresion_binaria.py](./Proyecto/regresion_binaria.py)
- [Proyecto/mlp.py](./Proyecto/mlp.py)
- [Proyecto/dnn.py](./Proyecto/dnn.py)
- [Proyecto/umbral.py](./Proyecto/umbral.py)
- [Proyecto/analisis_umbral.py](./Proyecto/analisis_umbral.py)
- [Proyecto/analisis_diferencias_clases.py](./Proyecto/analisis_diferencias_clases.py)
- [Proyecto/importancia.py](./Proyecto/importancia.py)

Nota importante:

- El experimento de `Random Forest` sí está desarrollado en el trabajo principal, pero en este repositorio no aparece un script `.py` dedicado para ese modelo; su flujo principal quedó documentado en el notebook.

## Resultados reportados en el notebook

Sobre datos balanceados, el resumen comparativo del notebook reporta:

| Modelo | Accuracy | F1 No Anómala | F1 Anómala |
| --- | ---: | ---: | ---: |
| Árbol de Decisión | 0.8565 | 0.86 | 0.85 |
| Random Forest | 0.8970 | 0.90 | 0.89 |
| MLP | 0.6612 | 0.72 | 0.56 |
| Regresión Logística | 0.7840 | 0.77 | 0.80 |
| DNN | 0.7999 | 0.78 | 0.81 |

Conclusión principal del proyecto:

- `Random Forest` fue el mejor modelo global en el conjunto balanceado.
- `Árbol de Decisión` también mostró un rendimiento competitivo.
- `DNN` y `Regresión Logística` ofrecieron buen recall para la clase anómala.
- `MLP` fue el modelo más débil entre los comparados.

## Resultados destacados por modelo

### Regresión Logística

Conjunto balanceado:

- `Accuracy`: `0.7840`
- mejor configuración: `C=0.1`, `solver=liblinear`, `class_weight=balanced`, `max_iter=500`

Ajuste de umbral:

- umbral óptimo: `0.512`
- `Recall clase 0`: `0.7253`
- `Recall clase 1`: `0.8421`

### DNN

Conjunto balanceado:

- `Accuracy`: `0.7999`
- mejor configuración: `3` capas, `10` neuronas, `dropout=0.2`, `lr=0.0005`, `epochs=20`, `batch_size=32`

Ajuste de umbral:

- umbral óptimo: `0.015`
- `Recall clase 0`: `0.6553`
- `Recall clase 1`: `0.9231`

### Random Forest

En los resultados del proyecto, aparece como el mejor modelo global en datos balanceados:

- `Accuracy`: `0.8970`
- `Precisión anómala`: `0.93`
- `Recall anómala`: `0.86`
- `F1 anómala`: `0.89`

## Cómo reproducir

### 1. Instalar dependencias

No hay un `requirements.txt` en la raíz, pero por el notebook y los scripts se necesitan al menos:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib tensorflow scikeras scipy
```

### 2. Generar datasets procesados

Sin escalado:

```bash
python Proyecto/limpieza_y_balanceo.py
```

Con escalado:

```bash
python Proyecto/limpieza_y_balanceoescalado.py
```

### 3. Entrenar modelos

Ejemplos:

```bash
python Proyecto/arboles_de_decision.py
python Proyecto/regresion_binaria.py
python Proyecto/dnn.py
```

Para `MLP`, las funciones están implementadas en [Proyecto/mlp.py](./Proyecto/mlp.py), pero su ejecución principal está integrada en el notebook.

### 4. Ajustar umbral de decisión

Para modelos con probabilidades:

```python
import umbral as u

u.main(
    modelo_path="modelo_logistic_regression_binaria.pkl",
    test_csv="Cloud_Anomaly_Dataset_Test_Balanceado_E.csv"
)
```

## Estructura del repositorio

```text
.
├── README.md
└── Proyecto/
    ├── TrrabajoFinalIA.ipynb
    ├── Cloud_Anomaly_Dataset.csv
    ├── limpieza_y_balanceo.py
    ├── limpieza_y_balanceoescalado.py
    ├── arboles_de_decision.py
    ├── regresion_binaria.py
    ├── mlp.py
    ├── dnn.py
    ├── umbral.py
    ├── analisis_umbral.py
    ├── analisis_diferencias_clases.py
    ├── importancia.py
    └── modelos y datasets generados
```

## Observaciones
- Algunos scripts esperan ejecutarse desde la carpeta `Proyecto/`, porque usan rutas relativas a los CSV generados.
- Hay archivos auxiliares de ejemplos académicos en `Proyecto/tree/`, `Proyecto/regression/` y `Proyecto/Random_Forest/` que no forman parte central del pipeline de detección de anomalías cloud.
