#Comparing resultos of Naive Bayes, Support Vector Machine and Random Forest algorithms in a classification task for a credit analysis application with and without feature selection

# For this investigation, we´re going to use a generic spreedcheet containing credit information from several applicants called "credit"
#The File is uploades

library(e1071)
library(FSelector)
library(randomForest)

#Amazenaremos nossos dados numa matriz a partir do seguinte comando:
credito = read.csv(file.choose(), se=",", header = T)
dim(credito)
#[1] 1000   21

# Ao inspecionarmos a matriz podemos ver que ela tem 20 atributos (colunas) e 1000 instâncias

#Iremos agora utilizar a técnica de hold-out para evitar um super ajuste dividindo nossa base de dados
#em duas partes: Uma reservada para treino (~80%) e a outra para teste (~20%).

amostra = sample(2,1000,replace=T,prob = c(0.8,0.2))
credito_treino = credito[amostra==1,]
credito_teste = credito[amostra==2,]

#Apos esse processo, temos duas novas matrizes (credito_treino e credito_teste) com as seguintes dimensões
dim(credito_treino)
#[1] 790  20
dim(credito_teste)
#[1] 210  20

#Using Naive Bayes
modelo = naiveBayes(CLASS ~ . , credito_treino)

#Faremos agora o teste do nosso modelo utilizando a função 'predict' e os dados de "credito_teste"
pred = predict(modelo, credito_teste)

#Com nossa predição armazenada em "pred", precisamos agora fazer nossa matriz de confusão para avaliar o desempenho do nosso modelo.

confusao = table(credito_teste$class, pred)
confusao
#      pred
#      bad good
# bad   33   33
# good  23  120

#Vamos agora analizar nossa margem de acerto invocando a variável TP

NB_HitRate = (confusao[1] + confusao[4])/sum(confusao)
NB_HitRate
#[1] 0.7320574

#Feature Selection

random.forest.importance(class ~ . , credito)

#                       attr_importance
#checking_status              47.793699
#duration                     25.441410
#credit_history               20.729415
#purpose                      13.496853
#credit_amount                19.482540
#savings_status               14.122255
#employment                    8.183450
#installment_commitment        9.399033
#personal_status               4.033948
#other_parties                11.115192
#residence_since               5.946515
#property_magnitude            9.670242
#age                          11.620429
#other_payment_plans          12.551811
#housing                       8.223140
#existing_credits              6.532143
#job                           4.805488
#num_dependents                2.932871
#own_telephone                 4.852106
#foreign_worker                3.653385

# Feature selection using only the most 5 important features

modelo = naiveBayes(class ~ checking_status + duration + credit_history + credit_amount + savings_status + purpose , credito_treino)
pred = predict(modelo, credito_teste)
confusao = table(credito_teste$class, pred)
confusao

#      pred
#      bad good
# bad   21   45
# good  15  128

NBFS_HitRate = (confusao[1] + confusao[4])/sum(confusao)
NBFS_HitRate
#[1] 0.7129187

#Comments on the result

#SVM
credito = read.csv(file.choose(), se=",", header = T)
amostra = sample(2,1000,replace=T,prob = c(0.8,0.2))
credito_treino = credito[amostra==1,]
credito_teste = credito[amostra==2,]

dim(credito_treino)
#[1] 805  21
dim(credito_teste)
#[1] 195  21

modelo = svm(class ~ . , credito_treino)
pred = predict(modelo, credito_teste)
confusao = table(credito_teste$class, pred)
confusao
#      pred
#      bad good
# bad   15   35
# good   9  136

SVM_HitRate = (confusao[1] + confusao[4])/sum(confusao)
SVM_HitRate
#[1] 0.774359

# Feature selection using only the most 5 important features

modelo = svm(class ~ checking_status + duration + credit_history + credit_amount + savings_status + purpose , credito_treino)
pred = predict(modelo, credito_teste)
confusao = table(credito_teste$class, pred)
confusao

#      pred
#      bad good
# bad   16   34
# good   8  137

SVMFS_HitRate = (confusao[1] + confusao[4])/sum(confusao)
SVMFS_HitRate

#[1] 0.7846154

#Comments on the result

# Random Forest

credito = read.csv(file.choose(), se=",", header = T)
amostra = sample(2,1000,replace=T,prob = c(0.8,0.2))
credito_treino = credito[amostra==1,]
credito_teste = credito[amostra==2,]

dim(credito_treino)
#[1] 802  21
dim(credito_teste)
#[1] 198  21

modelo = randomForest(class ~ . , credito_treino)
pred = predict(modelo, credito_teste)
confusao = table(credito_teste$class, pred)
confusao
#      pred
#      bad good
# bad   22   31
# good  15  130

RF_HitRate = (confusao[1] + confusao[4])/sum(confusao)
RF_HitRate
#[1] 0.7676768

# Feature selection using only the most 5 important features

modelo = randomForest(class ~ checking_status + duration + credit_history + credit_amount + savings_status + purpose , credito_treino)
pred = predict(modelo, credito_teste)
confusao = table(credito_teste$class, pred)
confusao

#      pred
#      bad good
# bad   26   27
# good  27  118

RFFS_HitRate = (confusao[1] + confusao[4])/sum(confusao)
RFFS_HitRate

#[1] 0.7272727

#Comments on the result

#Final Comments