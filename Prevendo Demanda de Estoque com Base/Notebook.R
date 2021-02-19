# Prevendo Demanda de Estoque com Base em Vendas

# Dataset disponível em: 
# https://www.kaggle.com/c/grupo-bimbo-inventory-demand

# Definindo diretório de trabalho
setwd("E:/Files/OneDrive - fgandrade.com/Doc/Course/DS/DSA/FormacaoCientistaDeDados/BigDAnalyticsRAzureML/20.ProjComFeedback/Projetos-1-2/02")
getwd()

# Carregando Dataset
library(data.table)
dataset <- fread(file.choose(), header = TRUE, col.names=c("id", "Semana", "Agencia_ID", "Canal_ID", "Ruta_SAK", "Cliente_ID", "Producto_ID", "Venta_uni_hoy", "Venta_hoy", "Dev_uni_proxima", "Dev_proxima", "Demanda_uni_equil")) # Escolha train.csv

# Exibindo informações do Dataset
dim(dataset)
str(dataset)
class(dataset)
sum(is.na(dataset)) # Checando valores missing / nulos
head(dataset, 10)
View(dataset[1:100,], title = 'Resumo Dataset')

# Visualizando correlação entre variáveis numéricas
library(corrplot)
num_vars <- sapply(dataset, is.numeric)
cor_matrix <- cor(dataset)
cor_matrix
corrplot(cor_matrix, main = '\n\n Gráfico de Correlação das Variáveis Numéricas', method = "number")
corrplot(cor_matrix, method = "color")
### Conforme observamos, as variáveis 'Venta_uni_hoy' e 'Venta_hoy' possuem uma alta correlação positiva com a variável target('Demanda_uni_equil') ###

# Gerando um dataframe de amostra a partir do original(
sample_df <- dataset[1:500000,]

# Gerando outro dataframe com apenas as variáveis de alta correlação
high_cor <- sample_df
high_cor[,c('Semana', 'Ruta_SAK', 'Cliente_ID', 'Canal_ID',
            'Agencia_ID','Producto_ID', 'Dev_uni_proxima' ,'Dev_proxima' )] <- NULL
View(high_cor)

## Análise Exploratória dos dados
# Médias de Tendência Central
summary(sample_df$Demanda_uni_equil)

quantile(sample_df$Demanda_uni_equil)
IQR(sample_df$Demanda_uni_equil) # Q3 - Q1
diff(range(sample_df$Demanda_uni_equil))
var(sample_df$Demanda_uni_equil)
sd(sample_df$Demanda_uni_equil)

## Plotando gráficos
# ScatterPlot
plot(x = sample_df$Venta_uni_hoy, y = sample_df$Demanda_uni_equil,
     main = "Scatterplot - Sales Unit x Adjusted Demand ",
     xlab = "Vendas unitárias", ylab = "Demanda Ajustada")

pairs(sample_df[, c('Venta_uni_hoy','Venta_hoy','Demanda_uni_equil')])

# Histograma
hist(sample_df[, Demanda_uni_equil], main = 'Quantidade de Demanda por Estoque', xlab = 'Demanda de Estoque', ylab = "Frequência", labels = T, breaks = c(0, 500, 1000, 1500, 2000))

# Boxplot
boxplot(sample_df[, c('Venta_uni_hoy','Venta_hoy','Demanda_uni_equil')], main = "Demanda por Estoque")
 
## Normalização das variáveis

# Criando função de normalização
normalize <- function(df, vars){
  for (var in vars) {
    df[[var]] <- scale(df[[var]], center = T, scale = T)
  }
  return(df)
}

# Normalizando
num_vars <- c('Venta_uni_hoy','Venta_hoy')
norm_df <- normalize(high_cor, num_vars)
View(norm_df)

## Separando Dados de Treino e Teste  
# Dataset Normalizado
set.seed(123)
lines <- sample(1:nrow(norm_df), 0.7 * nrow(norm_df))
train_norm <- norm_df[lines,]
test_norm <- norm_df[-lines,]

# Dataset de amostra
lines <- sample(1:nrow(sample_df), 0.7 * nrow(sample_df))

train_sample <- sample_df[lines,]
test_sample <- sample_df[-lines,]

# Conferindo se a divisão entre dados de treino e teste estão corretas
dim(train_norm); dim(test_norm)
dim(train_sample); dim(test_sample)


## 1- Utilizando o Algoritmo Regressão Linear
# Criando modelos
model_v1 <- lm(Demanda_uni_equil ~ . , data = train_norm)
model_v2 <- lm(Demanda_uni_equil ~ . , data = train_sample)

# Resumo dos modelos
summary(model_v1)
summary(model_v2)

# Prevendo modelos
pred_v1 <- predict(model_v1, test_norm)
pred_v2 <- predict(model_v2, test_sample)

class(pred_v1)
class(pred_v2)

View(pred_v1)
View(pred_v2)

# Acurácia dos modelos
actual_pred_v1 <- data.frame(cbind(actuals=test_norm$Demanda_uni_equil, predicteds= pred_v1))
actual_pred_v2 <- data.frame(cbind(actuals=test_sample$Demanda_uni_equil, predicteds=pred_v2))

View(actual_pred_v1)
View(actual_pred_v2)

# Arredondando valores previstos(Como podem notar, estão em formato decimal)
actual_pred_v1$predicteds <- round(actual_pred_v1$predicteds, 0)
actual_pred_v2$predicteds <- round(actual_pred_v2$predicteds, 0)

View(actual_pred_v1)
View(actual_pred_v2)

cor_acc_v1 <- cor(actual_pred_v1)
head(cor_acc_v1)

cor_acc_v2 <- cor(actual_pred_v2)
head(cor_acc_v2)


## 2- Utilizando o Algoritmo Decision Tree
library(party)

# Criando modelos
model_v3 <- ctree(Demanda_uni_equil ~ ., train_norm)
model_v4 <- ctree(Demanda_uni_equil ~ ., train_sample)

# Prevendo com dados de teste
pred_v3 <- predict(model_v3, test_norm)
pred_v4 <- predict(model_v4, test_sample)

View(pred_v3)
View(pred_v4)

# Arredondando valores previstos(Como podem notar, estão em formato decimal)
pred_v3 <- round(pred_v3, 0)
pred_v4 <- round(pred_v4, 0)

View(pred_v3)
View(pred_v4)

# Matriz de Confusão
table(pred_v3, test_norm$Demanda_uni_equil)
table(pred_v4, test_sample$Demanda_uni_equil)

# Descobrindo acurácia dos modelos
sum(pred_v3 == test_norm$Demanda_uni_equil)/length(test_norm$Demanda_uni_equil)*100
sum(pred_v4 == test_sample$Demanda_uni_equil)/length(test_sample$Demanda_uni_equil)*100


# Conclusão, podemos usar o segundo algoritmo testado, o de Decision Tree, ambos tiveram uma alta acurácia em seus modelos, 97% e 99% respectivamente