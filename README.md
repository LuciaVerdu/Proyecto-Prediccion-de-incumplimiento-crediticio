# Predicción de Incumplimiento Crediticio

Proyecto ABP desarrollado para el módulo **Analista de Datos II** (Ciencia de Datos II y Estadística y Exploración de Datos II) de la **Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial** — Instituto Superior Politécnico de Córdoba (ISPC).

## Integrantes

- Negro, Clarisa
- Verdú, Lucía
- Pucheta, Matías
- Virga, Camila

## Docentes

- Barbero, Maciel
- Pratta, Nahuel

## Descripción del problema

Las entidades financieras enfrentan el desafío de evaluar la capacidad de pago de solicitantes de crédito, buscando minimizar las pérdidas por morosidad sin restringir excesivamente el otorgamiento a clientes solventes. Este proyecto desarrolla una **Prueba de Concepto (PoC)** basada en Machine Learning para estimar la probabilidad de incumplimiento (`TARGET`) al momento de la solicitud.

**Pregunta de negocio:** ¿Cómo podemos identificar anticipadamente clientes con mayor probabilidad de incumplir un crédito para mejorar la toma de decisiones del área de riesgo financiero?

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle). Se utiliza el archivo `application_train.csv`, que contiene información demográfica, laboral, económica y del préstamo solicitado, junto con la variable objetivo `TARGET` (0 = cliente cumplidor, 1 = default).

## Notebook

📓 [Ver notebook en Google Colab](https://colab.research.google.com/drive/1HMh71TmK13NA1yqRD9L6olMQakDYNoGO?usp=sharing)

## Estado del proyecto

| Evidencia | Contenido | Estado |
|---|---|---|
| **Evidencia 1** | Comprensión del negocio, comprensión del dataset y Análisis Exploratorio de Datos (EDA) | ✅ Completa |
| **Evidencia 2** | Limpieza y preparación de datos, Ingeniería de características y Transformación de variables | ✅ Completa |
| **Evidencia 3** | Revisión de multicolinealidad (VIF), Entrenamiento de modelos, Evaluación, Optimización y Conclusiones de negocio | ✅ Completa |

---

## Contenido detallado por Evidencias

### **Evidencia 1**
1. **Comprensión del negocio:** Análisis del contexto financiero, identificación de las necesidades del stakeholder (Área de Riesgo Financiero) y definición del Marco de Valor Esperado (costo asimétrico de Falsos Negativos vs. Falsos Positivos).
2. **Comprensión del dataset:**
   - 2.0 Configuración del entorno (`pandas`, `numpy`, `matplotlib`, `seaborn`).
   - 2.1 Importación de datos (`application_train.csv` y diccionario de variables).
   - 2.2 Inventario estructural (246.008 registros en train × 122 columnas).
   - 2.3 Separación del test set en cuarentena (20% mantenido en reserva con `stratify=TARGET`).
   - 2.4 Diccionario de variables curado en 10 bloques temáticos de negocio.
   - 2.5 Análisis de valores faltantes (concentrados en un 68-70% en el bloque de vivienda).
   - 2.6 Distribución de la variable objetivo (desbalance de clases: 91.93% cumplidores vs. 8.07% default).
3. **Análisis Exploratorio de Datos (EDA):**
   - 3.0 Preparación de variables (`YEARS_BIRTH` y `YEARS_EMPLOYED` en años positivos).
   - 3.1 Descripción de los datos por bloques temáticos.
   - 3.2 Distribuciones de las variables (asimetría en montos monetarios).
   - 3.3 Relación de las variables con `TARGET` (análisis por tasa de default).
   - 3.4 Correlaciones (identificación de predictores clave como `EXT_SOURCE_1/2/3`).
   - 3.5 Problemas de calidad de datos (detección del valor centinela `365243` en antigüedad laboral y redundancias).
4. **Conclusiones del EDA:** Resumen de patrones de riesgo y hoja de ruta para la preparación de datos.

---

### **Evidencia 2**
5. **Limpieza y Preparación de Datos:**
   - 5.1 Tratamiento del valor centinela `365243` en `DAYS_EMPLOYED` (conversión a `NaN` y creación del flag `SIN_EMPLEO_REGISTRADO`).
   - 5.2 Eliminación de variables redundantes (`FLAG_EMP_PHONE` y `ORGANIZATION_TYPE`).
   - 5.3 Reducción del bloque de vivienda (47 variables conservando únicamente la variante `_MEDI`).
   - 5.4 Eliminación de filas con categorías anecdóticas (`CODE_GENDER = 'XNA'`, `NAME_FAMILY_STATUS = 'Unknown'`).
   - 5.5 Eliminación de variables casi constantes (`FLAG_DOCUMENT_*` y `FLAG_MOBIL`) y del identificador `SK_ID_CURR`.
   - 5.6 Tratamiento del valor extremo en `AMT_INCOME_TOTAL` (capping al percentil 99).
   - 5.7 Exclusión de la variable edad (`YEARS_BIRTH` / `DAYS_BIRTH`) por criterio de equidad (*Fairness*) y reserva en `train_audit_fairness`.
   - 5.8 Imputación de `EXT_SOURCE_1/2/3` mediante Iterative Imputer (MICE) y creación de missings flags (`_FALTABA`).
   - 5.9 Tratamiento de nulos en el bloque de vivienda con la bandera `SIN_DATO_VIVIENDA` e imputación por mediana.
   - 5.10 Manejo de categorías en `OCCUPATION_TYPE` ("Sin_dato") y `OWN_CAR_AGE` (imputado con 0).
   - 5.11 Verificación final y tratamiento de nulos residuales.
6. **Ingeniería de Características (Feature Engineering):**
   - 6.1 y 6.2 Ratios de capacidad de pago (`CUOTA_INGRESO_RATIO`, `CREDITO_INGRESO_RATIO`).
   - 6.3 Score combinado `EXT_SOURCE_PROMEDIO`.
7. **Transformación de Variables:**
   - 7.1 Transformación logarítmica `log(1+x)` de variables monetarias.
   - 7.2 y 7.3 Codificación de variables categóricas (`OneHotEncoder` con `drop='first'` y `OrdinalEncoder` para educación).
   - 7.4 Escalado de variables numéricas con `StandardScaler`.
- **Síntesis de la Evidencia 2:** Balance de transformación del dataset (`train_clean` listo con 124 columnas y sin nulos).

---

### **Evidencia 3**
8. **Revisión de multicolinealidad antes del modelado:**
   - 8.1 Factor de Inflación de la Varianza (VIF) en numéricas (eliminación de `CNT_FAM_MEMBERS`, `EXT_SOURCE_PROMEDIO`, `OBS_60_CNT_SOCIAL_CIRCLE`, `AMT_GOODS_PRICE_LOG` y `REGION_RATING_CLIENT`).
   - 8.2 VIF en variables categóricas / dummies.
   - 8.3 Conclusiones y selección final de características.
9. **División del Conjunto de Datos (Train / Test Split):** Partición estratificada 80/20 sobre `train_clean` para evaluación interna (49.201 clientes en `X_test`).
10. **Entrenamiento del Modelo Baseline - Regresión Logística:** Ajuste del modelo de referencia con `class_weight='balanced'`.
11. **Evaluación del Modelo Baseline y Análisis de Umbrales:**
    - 11.1 Qué mide el AUC-ROC y justificación de su elección frente a métricas sensibles al desbalance.
    - 11.2 Análisis de umbrales (comparación de umbral estándar 0.50 vs. conservador 0.30/0.35).
    - 11.3 Interpretación de resultados técnicos.
    - 11.4 Criterios de negocio para la selección del umbral óptimo según el costo de error.
12. **Optimización del Modelo Principal (Regresión Logística Optimizada):** Búsqueda de hiperparámetros con `GridSearchCV` (`C=0.01`, penalización L2) y prueba de modelo de contraste (`LightGBM`).
13. **Interpretabilidad y Coeficientes del Modelo Final:** Análisis del efecto estandarizado de los coeficientes (impacto de `EXT_SOURCE_*`, ratios financieros y estabilidad laboral).
14. **Cuadro Comparativo de Modelos y Decisión Técnica:** Evaluación comparativa del desempeño y selección del modelo ABP por su equilibrio entre capacidad predictiva y auditabilidad.
15. **Conclusiones de Negocio y Recomendaciones Financieras:**
    - Revisión de equidad por género y edad (auditoría de sesgo indirecto).
    - Cierre del proyecto (síntesis CRISP-DM, limitaciones y recomendaciones de puesta en producción).

---

## Cuadro Comparativo de Modelos (Evidencia 3)

| Modelo / Configuración | ROC-AUC | Recall Default (Captura Morosos) | Precision Default | % Clientes Rechazados |
|---|:---:|:---:|:---:|:---:|
| **Baseline: Regresión Logística (Umbral 0.50)** | 0.7392 | 67.7% | 15.6% | 35.0% |
| **Baseline: Regresión Logística (Umbral 0.30)** | 0.7392 | 91.0% | 11.0% | 68.0% |
| **Modelo Final: Logística Optimizada (Umbral 0.50)** | **0.7391** | **68.0%** | **15.6%** | **35.1%** |
| **Modelo Final: Logística Optimizada (Umbral 0.35)** | **0.7391** | **85.8%** | **11.6%** | **59.7%** |
| **Contraste: LightGBM (Umbral 0.50)** | 0.7448 | 66.8% | 16.0% | 33.7% |
| **Contraste: LightGBM (Umbral 0.35)** | 0.7448 | 85.4% | 12.0% | 57.0% |

---

## Estructura del repositorio
├── Grupo_7_Modulo_Analista_2.ipynb   # Notebook ejecutable con las 3 Evidencias
├── Grupo_7_Informe_Final.pdf         # Memoria técnica e informe ejecutivo del proyecto
└── README.md                          # Documentación general del repositorio
