# Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile

# Dataset disponível em: 
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

# Definindo diretório de Trabalho
setwd("E:/Files/OneDrive - fgandrade.com/Doc/Course/DS/DSA/FormacaoCientistaDeDados/BigDAnalyticsRAzureML/20.ProjComFeedback/Projetos-1-2/01")
getwd()

# Carregando dataset
df = read.csv('train_sample.csv', stringsAsFactors = FALSE)
View(df)
dim(df)
str(df)

# Tem valores nulos?
sum(is.na(df))

# Visualizando correlação entre variáveis
numeric_columns <- sapply(df, is.numeric)
data_cor <- cor(df[,numeric_columns])
head(data_cor)

# Plotando um corrplot
library(corrplot)
corrplot(data_cor, method = 'color')

# Scatterplot - Mostrando relacionamento entre as variáveis
pairs(df[c("ip", "app", "device","is_attributed")])

# Conforme apresentado, as variáveis que possuem maior correlação positiva com a variável target(is_attributed) são: ip e app

## Análise Exploratória dos Dados

# Resumo do Dataset
summary(df)

# Desvio Padrão
sd(df$is_attributed)
sd(df$ip)
sd(df$app)
sd(df$device)
sd(df$os)
sd(df$channel)

# variância
var(df$is_attributed)
var(df$ip)
var(df$app)
var(df$device)
var(df$os)
var(df$channel)

# Quartis
quantile(df$channel)
quantile(df$ip)
quantile(df$app)
quantile(df$device)
quantile(df$os)

# Plot
boxplot(df$ip, main = 'Endereço de IP de clique')
boxplot(df$app,main = 'ID do Aplicativo para Marketing')

hist(df$app, breaks = c(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600),  labels = T, xlab = "Id app", ylab = "Frequência", main = "Histograma de ID do Aplicativo")

# Cross tabulation
# Aqui iremos analisar se estão balanceados os resultados da nossa variável target
# ou seja, se estão na mesma proporção
table(df$is_attributed)

# Plotando a proporção do dataset desbalanceado
colors()
pie(table(df$is_attributed), col = c("darksalmon", "black"), main = "Proporção de Cliques" )

# Balanceando dataset
library(DMwR)
balanced_df <- read.csv('train_sample.csv', colClasses = c('is_attributed' = 'factor'))
str(balanced_df$is_attributed)
balanced_df <- SMOTE(is_attributed ~ . , balanced_df, perc.over = 600, perc.under = 100)

table(balanced_df$is_attributed)

# Plotando a proporção do dataset balanceado
pie(table(balanced_df$is_attributed), col = c("darksalmon", "black"), main = "Proporção de Cliques" )

# Criando função para normalização de variáveis numéricas
norm_func <- function(df, vars){
  for (var in vars) {
    df[[var]] <- scale(df[[var]], center = T, scale = T)
  }
  return(df)
}

# Normalizando
num_vars <- c('ip', 'app', 'device', 'os', 'channel')
normalized_df <- norm_func(balanced_df, num_vars)
View(normalized_df)

# Removendo variáveis que não serão úteis para o modelo
normalized_df$click_time <- NULL
normalized_df$attributed_time <- NULL
View(normalized_df)

# amostras de forma randomica
library(caTools)
amostra <- sample.split(normalized_df$is_attributed, SplitRatio = 0.70)
train_df <- normalized_df[amostra,]
test_df <- normalized_df[-amostra,]


# 1- Classificando com algoritmo Knn
library(class)
model_v1 <- knn(train_df, test_df, train_df$is_attributed, k=2, prob = TRUE)

# Matriz de Confusão
table(test_df$is_attributed, model_v1)

# Tabela cruzada: Dados Observados x Dados Previstos
library(gmodels)
CrossTable(x= test_df$is_attributed, model_v1, prop.chisq = FALSE)

# Analizando Acurácia do KNN
accuracy <- sum(model_v1 == test_df$is_attributed)/length(test_df$is_attributed)*100
accuracy


# 2- Classificando com algoritmo SVM
library(e1071)
model_v2 <- svm(is_attributed ~ . , data = train_df)

# Resumo do modelo
summary(model_v2)

# Prevendo com valores de teste
pred <- predict(model_v2, test_df)
View(pred) # Visualizando resultados previstos

# Matriz de confusão
table(pred, test_df$is_attributed)

# Analizando Acurácia do SVM
accuracy_2 <- sum(pred == test_df$is_attributed)/length(test_df$is_attributed)*100
accuracy_2


# 3 - Classificando com Decision Tree (Àrvore de Decisão)
library(party)
model_v3 <- ctree(is_attributed ~ . , train_df)
plot(model_v3, type = 'simple')

# Prevendo com dados de teste
pred_v3 <- predict(model_v3, test_df)
View(pred_v3)

# Matriz de Confusão
table(pred_v3, test_df$is_attributed) 

# Precisão da Àrvore
pred_train <- predict(model_v3, train_df)
tab1 <- table(Predicted = pred_train, Actual = train_df$is_attributed)
tab2 <- table(Predicted = pred_v3, Actual = test_df$is_attributed)
print(paste('Acurácia da Àrvore de Decisão é: ', sum(diag(tab2))/sum(tab2)))

# Analisando com um segundo modelo, mas somente com as variáveis mais relevantes para a target
model_v4 <- ctree(is_attributed ~ ip + app, train_df)

# Prevendo com dados de teste
pred_v4 <- predict(model_v4, test_df)

# Tabela cruzada: Dados Observados x Dados Previstos
CrossTable(x= test_df$is_attributed, pred_v4, prop.chisq = FALSE)

# Percentual de previsões corretas dos 3 modelos
mean(model_v1 == test_df$is_attributed)
mean(pred == test_df$is_attributed)
mean(pred_v3 == test_df$is_attributed)
mean(pred_v4 == test_df$is_attributed)

# Conclusão
# Para descobrir se determinado clique é fradulento ou não, assim resolvendo o problema de negócio que foi solicitado, será utilizado o algoritmo Decision Tree(Àrvore de Decisão) com os modelos model_v3 ou model_v4, pois foi o que teve maior acurácia(precisão) de acertos.
