# Predicción de Incumplimiento Crediticio

Proyecto ABP desarrollado para la materia **Ciencia de Datos II - Estadística y Exploración de Datos II**, Módulo Analista de Datos II, de la **Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial** - Instituto Superior Politécnico de Córdoba (ISPC).

## Integrantes

- Negro, Clarisa
- Verdú, Lucía
- Pucheta, Matías
- Virga, Camila

## Docentes

- Barbero, Maciel
- Pratta, Nahuel

## Descripción del problema

Las entidades financieras necesitan tomar decisiones de otorgamiento de crédito reduciendo el riesgo de pérdidas económicas. Este proyecto desarrolla una **Prueba de Concepto (PoC)** para estimar la probabilidad de que un solicitante de crédito incumpla sus pagos, comparando un modelo **Baseline (Regresión Logística)** contra un modelo **ABP (Regresión Logística Optimizada)** y un modelo de **Contraste (LightGBM)**.

**Pregunta de negocio:** ¿Cómo podemos identificar anticipadamente clientes con mayor probabilidad de incumplir un crédito para mejorar la toma de decisiones del área de riesgo financiero?

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle, 2018). Se utiliza el archivo `application_train.csv`, que contiene información demográfica, laboral, económica y del préstamo solicitado, junto con la variable objetivo `TARGET` (0 = cliente cumplidor, 1 = default).

## Notebook

📓 [Ver notebook en Google Colab](https://colab.research.google.com/drive/1HMh71TmK13NA1yqRD9L6olMQakDYNoGO?usp=sharing)

## Estado del proyecto

| Evidencia | Contenido | Estado |
|---|---|---|
| **Evidencia 1** | Comprensión del negocio, comprensión del dataset y análisis exploratorio (EDA) | ✅ Completa |
| **Evidencia 2** | Limpieza de datos, ingeniería de características y transformación de variables | ✅ Completa |
| **Evidencia 3** | Multicolinealidad (VIF), baseline, optimización ABP (GridSearchCV), contraste (LightGBM) y conclusiones | ✅ Completa |

---

## Contenido de la Evidencia 1

- **Comprensión del negocio:** contexto financiero, necesidad del stakeholder (Área de Riesgo Financiero) y Marco de Valor Esperado (costo de falsos negativos vs. falsos positivos).
- **Comprensión del dataset:** inventario estructural (246.008 filas × 122 columnas en train), separación estratificada de test en cuarentena (`stratify=TARGET`), diccionario de variables agrupado en 10 bloques temáticos, análisis de valores faltantes y distribución de la variable objetivo (desbalance de clases ~92% / ~8%).
- **Análisis Exploratorio de Datos (EDA):** preparación de variables (`YEARS_BIRTH`, `YEARS_EMPLOYED`), descripción por bloques, distribuciones, relación de cada variable con `TARGET`, análisis de correlaciones y problemas de calidad de datos.

### Principales hallazgos (Evidencia 1)
- Fuerte desbalance de clases (91.93% cumplidores vs. 8.07% default), lo que descarta el *accuracy* como métrica principal de evaluación.
- Los **scores externos** (`EXT_SOURCE_1/2/3`) son las variables con mayor capacidad de separación, aunque presentan nulos relevantes.
- La **edad** y la **antigüedad laboral** muestran una relación clara e inversa con el riesgo de incumplimiento (los más jóvenes presentan mayor tasa de mora).
- Se detectaron problemas clave a resolver: el valor centinela `365243` en `DAYS_EMPLOYED`, redundancia masiva en el bloque de vivienda (correlaciones > 0.96) y variables casi constantes.

---

## Contenido de la Evidencia 2

- **Limpieza de datos:** conversión definitiva del valor centinela de `DAYS_EMPLOYED` a `NaN` (creando la bandera `SIN_EMPLEO_REGISTRADO`), eliminación de variables redundantes (`FLAG_EMP_PHONE`, `ORGANIZATION_TYPE`), reducción del bloque de vivienda conservando únicamente la variante `_MEDI` (14 atributos), depuración de categorías anecdóticas y variables casi constantes, y recorte (*capping*) del valor extremo en `AMT_INCOME_TOTAL` al percentil 99.
- **Criterio de equidad (Fairness):** se excluyó la edad (`YEARS_BIRTH` / `DAYS_BIRTH`) y el género (`CODE_GENDER`) del pipeline de modelado e imputación, reservándolos en `train_audit_fairness` para auditoría de sesgos.
- **Imputación avanzadas e indicadores:** imputación de `EXT_SOURCE_1/2/3` mediante **Iterative Imputer (MICE)** conservando las banderas `_FALTABA`.
- **Ingeniería de características:** ratios de capacidad de pago (`CUOTA_INGRESO_RATIO`, `CREDITO_INGRESO_RATIO`) y score promedio `EXT_SOURCE_PROMEDIO`.
- **Transformación de variables:** transformación logarítmica `log(1+x)` en montos monetarios, codificación categórica (`OneHotEncoder` con `drop='first'` y Ordinal para educación) y escalado numérico con `StandardScaler`.

---

## Contenido de la Evidencia 3

- **Revisión de multicolinealidad (VIF):** cálculo del Factor de Inflación de la Varianza y eliminación de variables redundantes exactas (`CNT_FAM_MEMBERS`, `EXT_SOURCE_PROMEDIO`, `OBS_60_CNT_SOCIAL_CIRCLE`, `AMT_GOODS_PRICE_LOG` y `REGION_RATING_CLIENT`).
- **División de datos (Train / Test Split):** partición estratificada del 80/20 sobre `train_clean` para evaluación interna (49.201 clientes en `X_test`).
- **Entrenamiento Baseline:** Regresión Logística con `class_weight='balanced'` (ROC-AUC = 0.7392).
- **Evaluación y Análisis de Umbrales:** evaluación del impacto en negocio al modificar el umbral de decisión (estándar 0.50 vs. estricto 0.30/0.35).
- **Optimización ABP (GridSearchCV):** búsqueda de hiperparámetros (`C`, penalización L1/L2) obteniendo `C=0.01` y penalidad L2 como mejor configuración (ROC-AUC = 0.7391).
- **Modelo de Contraste:** entrenamiento de `LightGBM` (ROC-AUC = 0.7448) demostrando que la ganancia técnica (+0.006) no justifica perder la interpretabilidad y auditabilidad de la Regresión Logística.
- **Auditoría de Fairness y Recomendaciones:** evaluación de impacto dispar por género y tramos de edad.

### Cuadro comparativo final

| Modelo / Configuración | ROC-AUC | Recall Default (Captura Morosos) | Precision Default | % Clientes Rechazados |
|---|:---:|:---:|:---:|:---:|
| **Baseline: Regresión Logística (Umbral 0.50)** | 0.7392 | 67.7% | 15.6% | 35.0% |
| **Baseline: Regresión Logística (Umbral 0.30)** | 0.7392 | 91.0% | 11.0% | 68.0% |
| **Modelo Final ABP: Logística Optimizada (Umbral 0.50)** | **0.7391** | **68.0%** | **15.6%** | **35.1%** |
| **Modelo Final ABP: Logística Optimizada (Umbral 0.35)** | **0.7391** | **85.8%** | **11.6%** | **59.7%** |
| **Contraste: LightGBM (Umbral 0.50)** | 0.7448 | 66.8% | 16.0% | 33.7% |
| **Contraste: LightGBM (Umbral 0.35)** | 0.7448 | 85.4% | 12.0% | 57.0% |

---

## Estructura del repositorio
├── Modulo_Analista.ipynb        # Notebook completo: EDA + Limpieza + Feature Engineering + VIF + Modelos
├── Grupo7.pdf                  # Memoria técnica / Informe ejecutivo consolidado
└── README.md                    # Documentación general del proyecto
