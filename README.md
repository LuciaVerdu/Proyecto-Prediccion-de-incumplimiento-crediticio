# Predicción de Incumplimiento Crediticio

Proyecto ABP desarrollado para la materia **Ciencia de Datos II - Estadística y Exploración de Datos II**, Módulo Analista de Datos II, de la **Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial** - Instituto Superior Politécnico de Córdoba (ISPC).

## Integrantes

- Negro, Clarisa
- Verdú, Lucía
- Pucheta, Matías
- Virga, Camila

## Docentes

Barbero, Maciel y Pratta, Nahuel

## Descripción del problema

Las entidades financieras necesitan tomar decisiones de otorgamiento de crédito reduciendo el riesgo de pérdidas económicas. Este proyecto desarrolla una **Prueba de Concepto (PoC)** para estimar la probabilidad de que un solicitante de crédito incumpla sus pagos, utilizando un modelo de **regresión logística**.

**Pregunta de negocio:** ¿cómo podemos identificar anticipadamente clientes con mayor probabilidad de incumplir un crédito para mejorar la toma de decisiones del área de riesgo financiero?

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle, 2018). Se utiliza únicamente el archivo `application_train.csv`, que contiene información demográfica, laboral, económica y del préstamo solicitado, junto con la variable objetivo `TARGET` (0 = cliente cumplidor, 1 = default).

## Notebook

📓 [Ver notebook en Google Colab](https://colab.research.google.com/drive/1HMh71TmK13NA1yqRD9L6olMQakDYNoGO?usp=sharing)

## Estado del proyecto

| Evidencia | Contenido | Estado |
|---|---|---|
| **Evidencia 1** | Comprensión del negocio, comprensión del dataset y análisis exploratorio (EDA) | ✅ Completa |
| **Evidencia 2** | Limpieza de datos, ingeniería de características y transformación de variables | ✅ Completa |
| **Evidencia 3** | Modelo baseline y desarrollo de la solución | ⏳ Pendiente |

## Contenido de la Evidencia 1

- **Comprensión del negocio:** contexto financiero, necesidad del stakeholder (Área de Riesgo Financiero) y Marco de Valor Esperado (costo de falsos negativos vs. falsos positivos).
- **Comprensión del dataset:** inventario estructural (307.511 filas × 122 columnas), separación estratificada de train/test, diccionario de variables agrupado en 10 bloques temáticos, análisis de valores faltantes y distribución de la variable objetivo (desbalance de clases ~92% / ~8%).
- **Análisis Exploratorio de Datos (EDA):** distribuciones, relación de cada variable con `TARGET`, análisis de correlaciones y detección de multicolinealidad, y recopilación de problemas de calidad de datos a resolver en la próxima etapa (valores centinela, nulos no aleatorios, outliers, variables redundantes).

### Principales hallazgos (Evidencia 1)

- Fuerte desbalance de clases (11 clientes cumplidores por cada 1 en default), lo que descarta el *accuracy* como métrica principal de evaluación.
- Los **scores externos** (`EXT_SOURCE_1/2/3`) son las variables con mayor relación con el riesgo, aunque también las que más datos faltantes tienen.
- La **edad** y la **antigüedad laboral** muestran una relación clara e inversa con el riesgo de incumplimiento.
- El **perfil socioeconómico** (educación, tipo de ingreso, ocupación) genera diferencias de hasta 6 veces en la tasa de default entre categorías.
- Se detectaron problemas de calidad de datos a resolver antes de modelar: un valor centinela en `DAYS_EMPLOYED`, redundancia masiva en el bloque de variables de vivienda (63 pares de variables con correlación > 0.90), y variables casi constantes sin poder predictivo.

## Contenido de la Evidencia 2

- **Limpieza de datos:** tratamiento del valor centinela de `DAYS_EMPLOYED`, eliminación de variables redundantes, reducción del bloque de vivienda (47 → 12 variables), depuración de categorías anecdóticas y variables casi constantes, y tratamiento del outlier extremo en `AMT_INCOME_TOTAL` (capping al percentil 99).
- **Ingeniería de características:** dos ratios de capacidad de pago (`CUOTA_INGRESO_RATIO`, `CREDITO_INGRESO_RATIO`) y un score combinado (`EXT_SOURCE_PROMEDIO`) que sintetiza los tres puntajes externos.
- **Transformación de variables:** corrección logarítmica de variables monetarias, codificación de variables categóricas (One-Hot y Ordinal Encoding) y escalado con `StandardScaler`, ajustado exclusivamente sobre el conjunto de entrenamiento para evitar Data Leakage.

### Principales decisiones (Evidencia 2)

- **Imputación de `EXT_SOURCE_1/2/3`:** se descartó la imputación simple por mediana —por debilitar la variable más predictiva del dataset— en favor de **Iterative Imputer (MICE)**, usando como predictoras los otros scores externos y la antigüedad laboral.
- **Criterio de equidad (Fairness):** se excluyó la variable edad (`YEARS_BIRTH`) de todo el pipeline de modelado, incluso como predictora en pasos de imputación, para evitar discriminación sistemática hacia clientes jóvenes. Se reserva en un conjunto separado (`train_audit_edad`) para una auditoría de sesgos en la Evidencia 3.
- **Missing flags:** se conservaron indicadores binarios de ausencia de dato (`_FALTABA`, `SIN_EMPLEO_REGISTRADO`, `SIN_DATO_VIVIENDA`) allí donde el propio hecho de faltar el dato demostró tener relación con `TARGET`.

## Estructura del repositorio

```
├── Modulo_Analista.ipynb                  # Notebook: EDA + limpieza + feature engineering + transformación
├── Grupo7.pdf                             # Entrega acumulativa: portada + informes ejecutivos (Ev.1 y Ev.2)
│                                          # + anexos técnicos (Ev.1 y Ev.2) + recursos digitales
└── README.md
```

## Tecnologías utilizadas

- Python (pandas, numpy)
- Preprocesamiento y modelado: scikit-learn (`IterativeImputer`, `StandardScaler`)
- Visualización: matplotlib, seaborn
- Entorno: Google Colab

