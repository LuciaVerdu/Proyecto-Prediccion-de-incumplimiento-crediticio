# Predicción de Incumplimiento Crediticio

Proyecto ABP desarrollado para el módulo **Analista de Datos II** (Ciencia de Datos II y Estadística y Exploración de Datos II) de la **Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial** — Instituto Superior Politécnico de Córdoba (ISPC). Ciclo lectivo 2026.

## Integrantes

- Negro, Clarisa
- Verdú, Lucía
- Pucheta, Matías
- Virga, Camila

## Docentes

- Barbero, Maciel
- Pratta, Nahuel

## Descripción del problema

Las entidades financieras necesitan decidir, antes de otorgar un crédito, qué solicitantes tienen más probabilidad de no pagarlo. Aprobar a quien no paga genera una pérdida directa de capital; rechazar a quien sí pagaría implica perder un cliente rentable. Este proyecto desarrolla una **Prueba de Concepto (PoC)** basada en **regresión logística** para estimar la probabilidad de incumplimiento (`TARGET`) de cada solicitante al momento de la solicitud.

**Pregunta de negocio:** ¿Cómo podemos identificar anticipadamente clientes con mayor probabilidad de incumplir un crédito para mejorar la toma de decisiones del área de riesgo financiero?

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle). Se utiliza únicamente el archivo `application_train.csv` (307.511 solicitudes, 122 variables), que contiene información demográfica, laboral, económica y del préstamo solicitado, junto con la variable objetivo `TARGET` (0 = cliente cumplidor, 1 = default). Solo el **8,07%** de los clientes cae en default.

El archivo `application_test.csv` de Kaggle no incluye la variable `TARGET`, por lo que el conjunto de prueba se construye separando un 20% estratificado de `application_train.csv` al inicio del proyecto, antes de cualquier exploración.

## Notebook

📓 [Ver notebook en Google Colab](https://colab.research.google.com/drive/1Na-hfNsDOaOMaraMohwcpScZwlon3aFp)

## Resultados principales

| Modelo | ROC-AUC (prueba) | Variables | Recall default (umbral 0.08) | Precision default (umbral 0.08) | % clientes rechazados (umbral 0.08) |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline: Regresión Logística (C = 1.0) | 0.7464 | 126 | 68.3% | 15.7% | 35.1% |
| **Modelo ABP: Logística Optimizada (C = 0.1, sin montos redundantes)** | **0.7457** | **124** | **68.6%** | **15.8%** | **35.2%** |

- **AUC 0.7457:** ante un cliente moroso y uno pagador elegidos al azar, el modelo le asigna más riesgo al moroso en 75 de cada 100 casos.
- **Sin sobreajuste:** el AUC del modelo ABP es casi igual en entrenamiento (0.7450), validación cruzada (0.7435) y prueba (0.7457).
- **Umbral de decisión:** con el umbral por defecto (0.50) el modelo detecta menos del 1% de los morosos. Con 0.08 (tasa promedio de default) detecta el 68.6% y rechaza al 35.2% de los solicitantes; con 0.05, el 84.7% y el 55.7%.
- **Factores de riesgo:** los puntajes externos (`EXT_SOURCE_1/2/3`) son los predictores más fuertes; una cuota alta respecto del ingreso aumenta el riesgo; la antigüedad laboral y el nivel educativo lo reducen.
- **Equidad:** el modelo no usa edad ni género, pero la revisión por grupos muestra un sesgo indirecto por edad (rechaza al 59.3% de los menores de 30 años y al 18.1% de los mayores de 60).

## Metodología

El proyecto sigue la metodología **CRISP-DM** y utiliza exclusivamente técnicas del material de las Unidades I a V del módulo.

| Etapa | Técnicas | Unidad |
|---|---|---|
| Separación de datos | División estratificada 80/20 al inicio, antes del EDA (prevención del *data snooping*) | I, II, IV |
| Limpieza | Control de duplicados, tratamiento del valor centinela, imputación con mediana y moda, indicadores de faltante (*missing flags*) | II |
| Ingeniería de características | Ratios de capacidad de pago, transformación logarítmica de montos | III |
| Transformación | One-Hot Encoding, Ordinal Encoding, escalado estándar, todo ajustado solo con entrenamiento | II |
| Variables redundantes | Matriz de correlación, *heatmap* y redundancia por construcción | II, III, IV |
| Modelado | Regresión logística con regularización L2; búsqueda del hiperparámetro C con Grid Search y validación cruzada estratificada de 5 pliegues; selección de variables por parsimonia | III, IV, V |
| Evaluación | AUC-ROC, matriz de confusión, precision, recall, F1, análisis de umbrales, comparación entrenamiento vs. prueba, revisión de equidad | IV, V |

## Contenido del notebook

### Evidencia 1 — Comprensión del negocio, datos y EDA
1. **Comprensión del negocio:** contexto financiero, stakeholder (Área de Riesgo Financiero) y marco del valor esperado (costo de falsos negativos vs. falsos positivos).
2. **Comprensión del dataset:**
   - 2.0 Configuración del entorno.
   - 2.1 Importación de datos (`application_train.csv` y diccionario de variables).
   - 2.2 Inventario estructural (tipos de datos y variables).
   - 2.3 Separación del test set (20% estratificado, en cuarentena hasta la evaluación final).
   - 2.4 Diccionario de variables curado en bloques temáticos.
   - 2.5 Análisis de valores faltantes.
   - 2.6 Distribución de la variable objetivo (91,93% cumplidores vs. 8,07% default).
3. **Análisis Exploratorio de Datos (EDA):**
   - 3.0 Preparación de variables (edad y antigüedad laboral en años).
   - 3.1 Descripción de los datos por bloques.
   - 3.2 Distribuciones (histogramas; comparación de media y mediana).
   - 3.3 Relación de las variables con `TARGET` (tasa de default por grupo).
   - 3.4 Correlaciones.
   - 3.5 Problemas de calidad de datos (valor centinela `365243`, redundancias, valores extremos).
4. **Conclusiones del EDA.**

### Evidencia 2 — Preparación de los datos
5. **Limpieza y preparación:**
   - Control de registros duplicados.
   - 5.1 Valor centinela en `DAYS_EMPLOYED` (indicador `SIN_EMPLEO_REGISTRADO`).
   - 5.2 Eliminación de variables redundantes (`FLAG_EMP_PHONE`, `ORGANIZATION_TYPE`).
   - 5.3 Reducción del bloque de vivienda (se conserva la variante `_MEDI`).
   - 5.4 Eliminación de categorías anecdóticas (`CODE_GENDER = 'XNA'`, `NAME_FAMILY_STATUS = 'Unknown'`).
   - 5.5 Variables casi constantes e identificador `SK_ID_CURR`.
   - 5.6 Valor extremo de `AMT_INCOME_TOTAL` (se conserva y se trata con logaritmo en 7.1).
   - 5.7 Exclusión de edad y género por criterio de equidad (reservados para la auditoría).
   - 5.8 Imputación de `EXT_SOURCE_1/2/3` con la mediana e indicadores de faltante (`_FALTABA`).
   - 5.9 Nulos del bloque de vivienda (indicador `SIN_DATO_VIVIENDA` y mediana).
   - 5.10 `OCCUPATION_TYPE` ("Sin_dato") y `OWN_CAR_AGE`.
   - 5.11 Verificación final y nulos residuales.
6. **Ingeniería de características:** ratios `CUOTA_INGRESO_RATIO` y `CREDITO_INGRESO_RATIO`; score combinado `EXT_SOURCE_PROMEDIO`.
7. **Transformación de variables:**
   - 7.1 Transformación logarítmica `log(1 + x)` de los montos.
   - 7.2 y 7.3 One-Hot Encoding (una columna por categoría; las categorías nuevas se completan con la moda) y Ordinal Encoding del nivel educativo.
   - 7.4 Escalado con `StandardScaler`.
- **Síntesis de la Evidencia 2:** `train_clean` con 246.004 filas, 131 variables predictoras y sin nulos.

### Evidencia 3 — Modelado, evaluación y solución ABP
8. **Revisión de multicolinealidad:** matriz de correlación, *heatmap* y redundancia por construcción (se eliminan `CNT_FAM_MEMBERS`, `EXT_SOURCE_PROMEDIO`, `OBS_60_CNT_SOCIAL_CIRCLE`, `AMT_GOODS_PRICE_LOG` y `REGION_RATING_CLIENT`).
9. **Conjuntos de entrenamiento y evaluación:** `train_clean` completo para entrenar (246.004 clientes) y `test_df` para la evaluación final (61.503 clientes), preparado con la función `preparar_test` sin fuga de datos.
10. **Modelo baseline:** regresión logística con regularización L2, entrenada con las proporciones reales de default.
11. **Evaluación del baseline y análisis de umbrales:** AUC-ROC, matrices de confusión y umbrales 0.50, 0.08 y 0.05.
12. **Modelo ABP:**
    - 12.1 Optimización del hiperparámetro C (Grid Search con validación cruzada estratificada de 5 pliegues).
    - 12.2 Selección de variables: modelo sin montos redundantes (criterio de parsimonia).
    - 12.3 Entrenamiento y evaluación final (AUC en entrenamiento y en prueba).
    - 12.4 Interpretación de resultados.
13. **Interpretabilidad:** coeficientes de variables numéricas, indicadores y categorías.
14. **Cuadro comparativo y decisión técnica:** se elige el modelo ABP por su simplicidad e interpretabilidad con el mismo desempeño.
15. **Conclusiones de negocio:** recomendaciones de umbral, revisión de equidad por género y edad, y cierre del proyecto (limitaciones y recomendaciones para producción).

## Cómo ejecutar el notebook

1. Descargar `application_train.csv` desde [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) y subirlo a Google Drive junto con `HomeCredit_columns_description.csv` (incluido en este repositorio).
2. Abrir `ABP.ipynb` en Google Colab.
3. Ajustar la variable `ruta` de la sección 2.1 con la carpeta de Drive donde están los archivos.
4. Ejecutar **Entorno de ejecución → Ejecutar todas**. La ejecución completa tarda unos minutos.

Librerías utilizadas: `pandas`, `numpy`, `matplotlib`, `seaborn` y `scikit-learn`.

## Estructura del repositorio

```
├── ABP.ipynb                            # Notebook con las 3 evidencias y la solución ABP
├── HomeCredit_columns_description.csv   # Diccionario oficial de variables (Kaggle)
└── README.md                            # Documentación del repositorio
```

## Referencias

- Barbero, M. (2026). *Ciencia de Datos II: Unidades I a V* [Material de cátedra]. ISPC.
- Géron, A. (2017). *Hands-on machine learning with Scikit-Learn and TensorFlow*. O'Reilly Media.
- Montoya, A., inversion, KirillOdintsov, & Kotek, M. (2018). *Home Credit Default Risk* [Conjunto de datos]. Kaggle. https://kaggle.com/competitions/home-credit-default-risk
- Provost, F., & Fawcett, T. (2013). *Data science for business*. O'Reilly Media.
- Zheng, A., & Casari, A. (2018). *Feature engineering for machine learning*. O'Reilly Media.
