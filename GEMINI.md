> **Nota para Colaboradores (Humanos y de IA):** Este documento es la fuente de verdad y el plan estratégico para este proyecto. Su propósito es proporcionar un contexto completo y una hoja de ruta clara, permitiendo a cualquier agente entender los objetivos, las decisiones y los próximos pasos sin necesidad de revisar todo el historial de conversaciones.
> 
> **Regla de Modificación:** Antes de proponer una modificación a este archivo, el agente de IA debe leerlo en su estado actual para asegurar que sus cambios son contextuales y no sobrescriben información relevante de forma accidental.

# Propuesta y Plan de Acción: Proyecto Final de Machine Learning

**Materia:** Machine Learning
**Universidad:** Universidad de San Andrés (UdeSA)

---

## 1. Introducción: El Proyecto en Contexto

Este documento constituye la hoja de ruta estratégica para el proyecto final de la materia de Machine Learning. El requerimiento base del proyecto es desarrollar un modelo predictivo para estimar los precios de alquiler en el Área Metropolitana de Buenos Aires (AMBA). Sin embargo, la consigna incentiva explícitamente la exploración de extensiones que vayan más allá de la simple predicción, buscando extraer insights más profundos del dataset.

En este contexto, y tras una sesión de brainstorming, se ha decidido **pivotar el enfoque del proyecto**. En lugar de centrarse únicamente en la optimización de la precisión predictiva (un objetivo valioso pero estándar), el proyecto adoptará una narrativa más analítica y orientada a negocio. El objetivo no es solo predecir un precio, sino **utilizar el modelo de Machine Learning como una herramienta para una investigación cuantitativa** sobre la dinámica del mercado inmobiliario.

---

## 2. El Dataset: Alquileres en el AMBA

El proyecto se basa en el dataset `alquiler_AMBA_dev.csv`, que contiene publicaciones de alquileres en Mercado Libre Argentina durante 2021 y 2022.

### Características Principales:
- **Datos Geoespaciales:** Las propiedades están geolocalizadas mediante el centroide de polígonos de 200x200 metros, proporcionando coordenadas (`LONGITUDE`, `LATITUDE`) y datos de ubicación como provincia (`ITE_ADD_STATE_NAME`), ciudad (`ITE_ADD_CITY_NAME`) y barrio (`ITE_ADD_NEIGHBORHOOD_NAME`).
- **Features de la Propiedad:** Incluye características clave como superficie (`STotalM2`, `SConstrM2`), número de habitaciones (`Dormitorios`, `Banos`, `Ambientes`), antigüedad (`Antiguedad`) y cocheras (`Cocheras`).
- **Amenities:** Un amplio conjunto de variables booleanas que indican la presencia de amenities como `Pileta`, `Gimnasio`, `Seguridad`, `SUM`, etc.
- **Variable Objetivo:** La columna a predecir es `precio_pesos_constantes`, que representa el precio del alquiler ajustado por inflación, lo que permite una comparación temporal válida.

---

## 3. Título y Pregunta Central del Proyecto

**Título Propuesto:** "Valoración y Elasticidad de Amenities en el Mercado de Alquileres de AMBA: Un Estudio de Impacto por Zona Geográfica"

**Pregunta Central:** Más allá de predecir un precio, buscamos responder preguntas de negocio y de mercado:
- ¿Cuánto valor monetario o porcentual agrega realmente cada amenity (pileta, gimnasio, seguridad) a una propiedad?
- ¿Este "premio" por una amenity es universal, o depende críticamente de la zona geográfica (ej: Capital Federal vs. GBA Oeste)?
- ¿Qué características son las que más influyen en la formación de precios, y cuáles son prácticamente irrelevantes?
- ¿Qué amenity es la más relevante en cada barrio?

El objetivo es transformar el modelo predictivo en una **herramienta de consultoría** que permita cuantificar la "elasticidad" del precio respecto a las características de una propiedad.

---

## 4. La Estrategia: De la Predicción a la Explicación

El proyecto se ejecutará en dos fases principales, donde la primera fase (el modelo predictivo) es un medio para habilitar la segunda (el análisis explicativo).

### Fase 1: Construcción de un Modelo Predictivo Robusto

El objetivo de esta fase es entrenar un modelo de Machine Learning de alta precisión que entienda las complejas relaciones no lineales del mercado inmobiliario. Este modelo no es el fin, sino el **motor de nuestro análisis**.

**Pasos Clave:**
1.  **Preprocesamiento Avanzado:** Se implementará un pipeline de `scikit-learn` que ejecute las decisiones tomadas en el EDA:
    - Imputación de datos faltantes (mediana para numéricos, `False` para booleanos).
    - Creación de "flags" para valores imputados.
    - Clipping de outliers y transformaciones logarítmicas para normalizar distribuciones.
    - **Clustering Geoespacial (K-Means):** Se creará una feature `geo_cluster` a partir de las coordenadas para capturar la señal de ubicación de forma granular, reemplazando las columnas de `ciudad` y `barrio`.
    - One-Hot Encoding para todas las variables categóricas finales.
    - Estandarización de todas las features numéricas.
2.  **Entrenamiento del Modelo:** Se entrenará un modelo de Gradient Boosting (preferiblemente **LightGBM** o **XGBoost**), ya que son ideales para capturar las interacciones complejas que descubrimos en el EDA.
3.  **Evaluación:** Se evaluará el rendimiento del modelo usando métricas estándar (RMSE, R²) para asegurar que sus predicciones son fiables.

### Fase 2: Análisis Explicativo con SHAP

Una vez que tengamos un modelo preciso y confiable, lo usaremos como un "laboratorio" para entender el mercado.

**Pasos Clave:**
1.  **Cálculo de Valores SHAP:** Se utilizará la librería `shap` para calcular la contribución de cada feature a cada predicción individual que hace el modelo.
2.  **Análisis de Importancia Global:** Se generará un gráfico de barras de los valores SHAP promedio para obtener un ranking definitivo y robusto de la importancia de cada feature en la formación de precios.
3.  **Análisis de Elasticidad por Zona (El Entregable Principal):** Se agruparán los valores SHAP para responder nuestra pregunta central.
    - Se calculará el **impacto promedio en el precio (en pesos constantes o en `log(precio)`)** de tener `Pileta` para el subconjunto de datos de `Capital Federal`.
    - Se repetirá el cálculo para `GBA Norte`, `GBA Oeste`, etc.
    - Se hará lo mismo para otras amenities clave como `Seguridad`, `Gimnasio`, y para características como `Dormitorios`.
4.  **Visualización de Resultados:** Se creará un "dashboard" final (una serie de gráficos de barras o una tabla resumen) que muestre de forma clara y comparable el "premio" o "castigo" que el mercado le asigna a cada característica, desglosado por zona geográfica.

---

## 5. Ideas para Extensiones (Opcionales)

Si el tiempo lo permite, el proyecto podría extenderse con análisis aún más avanzados que aprovechan las herramientas ya construidas:

- **Idea Secundaria: "El ADN del Barrio"**: Utilizar técnicas de clustering sobre las características promedio de las propiedades de cada barrio para crear un "mapa de arquetipos" (ej: clusters de "barrios de lujo", "barrios familiares", "barrios de estudiantes").
- **Idea Experimental: "El Mapa del ADN Inmobiliario"**: Usar un Autoencoder Variacional (VAE) para reducir la dimensionalidad de todas las features a un espacio latente de 2D y visualizar la "estructura" completa del mercado inmobiliario en un solo gráfico.
