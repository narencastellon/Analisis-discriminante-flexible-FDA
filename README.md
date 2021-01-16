# Analisis-discriminante-flexible-FDA

## **6. Análisis discriminante flexible - FDA.**

La FDA es una extensión flexible de LDA que utiliza combinaciones no lineales de predictores como splines. La FDA es útil para modelar relaciones no normales o no lineales multivariadas entre variables dentro de cada grupo, lo que permite una clasificación más precisa.

## **Paso 1. Carga de paquetes R requeridos.**
Carga de paquetes R requeridos

`tidyverse` para una fácil visualización y manipulación de datos.
`caret` para un flujo de trabajo de aprendizaje automático (Machine Learning) sencillo.

```{r message=FALSE}
library(tidyverse)
library(caret)
library(klaR)
theme_set(theme_classic())
```


## **Paso 2. Preparando los datos.**

Usaremos el conjunto iris de datos, para predecir especies de iris basadas en las variables predictoras Sepal.Length, Sepal.Width, Petal.Length, Petal.Width.

El análisis discriminante puede verse afectado por la escala / unidad en la que se miden las variables predictoras. Generalmente se recomienda estandarizar / normalizar el predictor continuo antes del análisis.

**2.1. Divida los datos en entrenamiento y conjunto de prueba:**

```{r}
# Cargamos la data
data("iris")
# Dividimos la data para entrenamiento en un (80%) y para la prueba en un (20%)
set.seed(123)
training.samples <- iris$Species %>%
createDataPartition(p = 0.8, list = FALSE)
train.data <- iris[training.samples, ]
test.data <- iris[-training.samples, ]
```

**2. Normaliza los datos. Las variables categóricas se ignoran automáticamente.**

```{r}
# Estimar parámetros de preprocesamiento
preproc.param <- train.data %>% 
preProcess(method = c("center", "scale"))
# Transformar los datos usando los parámetros estimados
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)
```

# **Paso 3. Creación del Modelo QDA**
FDA se puede calcular usando la función `fda()`[paquete MASS]

```{r}
library(mda)
# Creando el modelo
modelfda <- fda(Species~., data = train.transformed)
modelfda
```
```{r}
#Coeficientes
coef(modelfda)
```


## **Paso 4. Gráficos de partición FDA**

El uso de la partimatfunción nuevamente proporciona una forma de graficar las funciones discriminantes cuadráticas. La única diferencia en el código de la sección LDA anterior es reemplazar `method="lda"`con `method="qda"`. Estos gráficos proporcionan una buena visualización de la diferencia entre las funciones lineales utilizadas en LDA y las funciones cuadráticas utilizadas en QDA. Nuevamente, las regiones coloreadas delinean cada área de clasificación. Se predice que cualquier observación que se encuentre dentro de una región sea de una clase específica. Cada gráfico también incluye la tasa de error aparente para esa vista de los datos.

```{r }
#partimat(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=train.transformed, method="fda")
plot(modelfda)
```

## **Paso 5: use el modelo para hacer predicciones FDA**

Una vez que hemos ajustado el modelo utilizando nuestros datos de entrenamiento, podemos usarlo para hacer predicciones sobre nuestros datos de prueba:

```{r}
# Haciendo predicciones
predicted.classes <- modelfda %>% predict(test.transformed)
```

## **Paso 6: evaluar el modelo FDA**

Podemos usar el siguiente código para ver para qué porcentaje de observaciones el modelo QDA predijo correctamente la Specie:

```{r warning=FALSE,message=FALSE}

# Precisión del Modelo
mean(predicted.classes == test.transformed$Species)
```

Resulta que el modelo predijo correctamente las especies para el 96.67% de las observaciones en nuestro conjunto de datos de prueba.
