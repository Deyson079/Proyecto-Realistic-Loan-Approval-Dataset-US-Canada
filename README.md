# Proyecto: Realistic Loan Approval Dataset | US & Canada

Modelo de clasificación binaria para predecir la aprobación o el rechazo de solicitudes de préstamo, usando un dataset realista de EE. UU. y Canadá.

**Autores:** Valeria Duque Moreno y Deyson Ortiz Restrepo

**Dataset:** [Realistic Loan Approval Dataset | US & Canada](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada) (Kaggle, autor: Parth Patel)

---

## Descripción del proyecto

Una entidad financiera (banco, fintech, cooperativa de crédito) necesita decidir de forma rápida y consistente si aprueba o no una solicitud de crédito, con base en el perfil demográfico, financiero y de comportamiento crediticio del solicitante.

Este proyecto aborda el problema como una tarea de **aprendizaje supervisado — clasificación binaria**, donde la variable objetivo `loan_status` indica si un préstamo fue aprobado (`1`) o rechazado (`0`).

Un modelo predictivo permite:
- Automatizar y acelerar el proceso de aprobación.
- Reducir el riesgo de otorgar créditos a solicitantes con alta probabilidad de incumplimiento.
- Aplicar criterios de decisión consistentes y auditables.
- Priorizar la revisión manual solo de los casos "límite" (probabilidad cercana a 0.5).

## Contenido del notebook

El notebook `Proyecto_Realistic_Loan_Approval_Dataset___US___Canada.ipynb` está organizado en las siguientes secciones:

1. **Descarga y descripción del dataset** — descarga automática desde Kaggle con `kagglehub` y descripción de variables.
2. **Análisis Exploratorio de Datos (EDA)** — distribución de variables numéricas y categóricas, outliers, correlaciones y relación con la variable objetivo.
3. **Preprocesamiento de datos** — train/test split estratificado (80/20), `OneHotEncoder` para variables categóricas y `StandardScaler` para variables numéricas, todo dentro de un `Pipeline` para evitar fuga de información.
4. **Validación cruzada (10 folds) y entrenamiento de modelos**, cada uno optimizado con `GridSearchCV`:
   - 4.1 Árboles de Decisión (`DecisionTreeClassifier`)
   - 4.2 Comité de máquinas / Bagging (con Regresión Logística y con Árboles)
   - 4.3 Random Forest
   - 4.4 AdaBoost
   - 4.5 XGBoost
5. **Comparación de modelos** — tabla y gráficos comparativos (accuracy, precision, recall, F1, ROC AUC, tiempo de entrenamiento) y curvas ROC.
6. **Guardado del mejor modelo** con `joblib` (pipeline completo: preprocesamiento + modelo).
7. **Conclusiones**.

## Variable objetivo

`loan_status`: variable binaria (`1` = préstamo aprobado, `0` = préstamo rechazado/negado). Distribución aproximada: 55% aprobados / 45% rechazados (leve desbalance de clases).

## Variables principales

| Variable | Descripción |
|---|---|
| `age` | Edad del solicitante |
| `occupation_status` | Situación laboral (Employed, Self-Employed, Student) |
| `years_employed` | Años en el empleo actual |
| `annual_income` | Ingreso anual bruto |
| `credit_score` | Puntaje de crédito (tipo FICO) |
| `credit_history_years` | Años de historial crediticio |
| `savings_assets` | Ahorros / activos líquidos |
| `current_debt` | Deuda actual total |
| `defaults_on_file` | Incumplimientos registrados (0/1) |
| `delinquencies_last_2yrs` | Pagos atrasados en los últimos 2 años |
| `derogatory_marks` | Marcas negativas en el reporte crediticio |
| `product_type` | Tipo de producto (Credit Card, Personal Loan, Line of Credit) |
| `loan_intent` | Propósito del préstamo |
| `loan_amount` | Monto solicitado |
| `interest_rate` | Tasa de interés |
| `debt_to_income_ratio` | Relación deuda/ingreso |
| `loan_to_income_ratio` | Relación monto préstamo/ingreso |

(`customer_id` se elimina por no aportar valor predictivo.)

## Resultados

| Modelo | ROC AUC (CV, 10-fold) | Accuracy (test) |
|---|---|---|
| Árbol de Decisión | 0.9525 | 89% |
| Bagging (mejor variante) | 0.9729 | 91% |
| Random Forest | 0.9756 | 91% |
| AdaBoost | 0.9837 | 93% |
| **XGBoost** | **0.9848** | **93%** |

**Modelo ganador: XGBoost**, con `learning_rate=0.1`, `max_depth=5`, `n_estimators=200`, `subsample=0.8`.

Se observa la jerarquía esperada: árbol individual < bagging (Random Forest / Bagging) < boosting (AdaBoost / XGBoost). El boosting supera al bagging porque ataca tanto el sesgo como la varianza, corrigiendo secuencialmente los errores del modelo anterior.

Las variables más influyentes según `feature_importances_` (Random Forest y XGBoost) son `credit_score` (≈0.21) y `debt_to_income_ratio` (≈0.17), seguidas por `interest_rate`, `credit_history_years` y `delinquencies_last_2yrs`.

## Modelo guardado

El mejor modelo (pipeline completo: `ColumnTransformer` + XGBoost) se serializa con `joblib` en el archivo:

```
mejor_modelo_loan_approval.joblib
```

Al cargarlo, recibe datos crudos del solicitante y se encarga internamente de codificar variables categóricas y escalar las numéricas antes de predecir — queda listo para producción sin reconstruir el preprocesamiento por separado.

```python
import joblib

modelo = joblib.load("mejor_modelo_loan_approval.joblib")
prediccion = modelo.predict(nuevos_datos)          # 0 = Rechazado, 1 = Aprobado
probabilidad = modelo.predict_proba(nuevos_datos)  # probabilidad de aprobación
```

## Requisitos

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
kagglehub
```

Instalación rápida:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib kagglehub
```

## Cómo ejecutar

1. Abrir el notebook `Proyecto_Realistic_Loan_Approval_Dataset___US___Canada.ipynb` en Jupyter o Google Colab.
2. Ejecutar las celdas en orden: la primera celda de código instala `kagglehub` y `xgboost`, y descarga automáticamente el dataset desde Kaggle (requiere credenciales de Kaggle configuradas).
3. El notebook entrena los 5 modelos con validación cruzada de 10 folds y `GridSearchCV`, genera la comparación final y guarda el mejor modelo en `mejor_modelo_loan_approval.joblib`.

## Conclusiones principales

- Es posible predecir automáticamente la aprobación de un préstamo con alta confiabilidad: **93% de acierto y ROC AUC de 0.984**.
- El modelo es interpretable y coherente con criterios reales de análisis de crédito (capacidad de pago e historial crediticio como factores dominantes).
- Los hallazgos del EDA (desbalance leve de clases, diferencias de escala, correlaciones fuertes de `credit_score` y ratios de endeudamiento) se confirmaron durante todo el proyecto.
- El modelo queda listo para un escenario de producción: automatizar decisiones de bajo riesgo y priorizar la revisión manual en los casos límite.
