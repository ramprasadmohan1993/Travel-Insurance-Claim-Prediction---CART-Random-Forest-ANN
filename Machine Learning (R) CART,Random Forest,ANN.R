getwd()
setwd("D:/R_wd")
########################################################################################################
#                                                                                                      #
#                         Purpose:       ML Project - CART , RF, ANN                                   #
#                                                                                                      #
#                         Author:        Ramprasad                                                     #
#                         Contact:       89396 ******                                                  #
#                         Client:        Great Learning                                                #
#                                                                                                      #
#                         Code created:  2020-08-16                                                    #
#                         Last updated:  2020-08-16                                                    #
#                         Source:        C:/Users/indway/Desktop                                       #
#                                                                                                      #                                                                                                    
########################################################################################################

###############################################################################################################
#                                                                                                             #
#                           First Part :  Decision Tree & Random Forest                                       #
#                                                                                                             #
###############################################################################################################


pacman::p_load(ggplot2,psych,readr,summarytools,DataExplorer,rpart,rpart.plot,rsample,vip,ROCR,randomForest,caret,MLmetrics)

insurance <- read.csv(file.choose(),header = TRUE)

head(insurance)

insurance$Claimed <- as.factor(insurance$Claimed)
str(insurance)


### --------------------------- Exploratory Data Analysis & Transformation ------------------------------------###

summarytools::view(dfSummary(insurance))
boxplot(insurance$Duration,horizontal = TRUE)

#### ---------------------------------Removing outliers & columns that are not required ------------------------------------------------####

maxm <- max(insurance$Duration)
minm <- min(insurance$Duration)
insurance <- insurance[!insurance$Duration==maxm,]
insurance <- insurance[!insurance$Duration==minm,]

boxplot(insurance$Duration,insurance$Age,insurance$Commision,insurance$Sales)

insurance$Channel <- NULL

summarytools::view(dfSummary(insurance))
DataExplorer::create_report(insurance)
funModeling::plot_num(insurance)
psych::pairs.panels(insurance,hist.col = "blue")
DataExplorer::plot_missing(insurance)


###############################################################################################################
#                           Split the data into training and testing model                                    #
###############################################################################################################

set.seed(123)
datasplitting <- initial_split(data = insurance,prop = 0.7,strata = "Claimed")

Insurance_train <- training(datasplitting)
Insurance_test <- testing(datasplitting)

head(Insurance_train)


###############################################################################################################
#                                      CART MODEL                                                             #
###############################################################################################################

x <- 9

cartctrl <- rpart.control(minsplit = x, minbucket = round(x/3),xval = 5,cp = 0)

CARTmodel <- rpart(formula = Claimed ~ .,data = Insurance_train,method = "class",control = cartctrl)

rpart::post(CARTmodel,file = "")

rpart.plot(CARTmodel)
rpart.plot::prp(CARTmodel)

vip::vip(CARTmodel,fill="blue")
caret::varImp(CARTmodel)

printcp(CARTmodel)
plotcp(CARTmodel,col = "red",upper = "splits")


prunedCART <- prune(CARTmodel,cp = 0.00463679, "CP")

rpart::post(prunedCART,file = "")
rpart.plot(prunedCART,type = 2,fallen.leaves = TRUE,extra = 104)
printcp(prunedCART)
plotcp(prunedCART)


rpart.plot::prp(prunedCART,type = 2,fallen.leaves = FALSE)

library(rattle)
fancyRpartPlot(prunedCART)

vip::vip(prunedCART,fill="dark blue")
caret::varImp(prunedCART)
###############################################################################################################
#                          Predict                                                                            #
###############################################################################################################

Newinsurancetrain <- Insurance_train
Newinsurancetest <- Insurance_test

Newinsurancetrain$predictedclass <- predict(prunedCART,Newinsurancetrain,type = "class")

Newinsurancetrain$probscores <- predict(prunedCART,Newinsurancetrain)[,2]

Newinsurancetest$predictedclass <- predict(prunedCART,Newinsurancetest,type = "class")

Newinsurancetest$probscores <- predict(prunedCART,Newinsurancetest)[,2]

###############################################################################################################
#                          Model Evaluation                                                                   #
###############################################################################################################

table(Newinsurancetrain$Claimed,Newinsurancetrain$predictedclass,dnn = c("Actual","Predicted"))

table(Newinsurancetest$Claimed,Newinsurancetest$predictedclass,dnn = c("Actual","Predicted"))

MLmetrics::Accuracy(Newinsurancetrain$predictedclass,Newinsurancetrain$Claimed)

MLmetrics::ConfusionMatrix(Newinsurancetrain$predictedclass,Newinsurancetrain$Claimed)

confusionMatrix(Newinsurancetrain$predictedclass,Newinsurancetrain$Claimed,positive = "Yes",mode = "everything")

confusionMatrix(Newinsurancetest$predictedclass,Newinsurancetest$Claimed,positive = "Yes",mode = "everything")

####------------------------------------------ROCR-----------------------------------------------------------###

library(ROCR)

ROCCART<-ROCit::rocit(score = Newinsurancetrain$probscores,class = Newinsurancetrain$Claimed)
plot(ROCCART)

predCART <- ROCR::prediction(Newinsurancetrain$probscores,Newinsurancetrain$Claimed)
perfCART <- performance(predCART, "tpr", "fpr")

ROCR::plot(perfCART, main = paste("AUC =", 0.786),colorize = TRUE,lwd = 2)
abline(0,1)

cartauc <- performance(predCART, "auc")

ROCCART1<-ROCit::rocit(score = Newinsurancetest$probscores,class = Newinsurancetest$Claimed)
plot(ROCCART1)

predCART1 <- ROCR::prediction(Newinsurancetest$probscores,Newinsurancetest$Claimed)
perfCART1 <- performance(predCART1, "tpr", "fpr")

ROCR::plot(perfCART1, main = paste("AUC =", 0.75),colorize = TRUE,lwd = 2)
abline(0,1)

cartauc1 <- performance(predCART1, "auc")

###############################################################################################################
#                                           RANDOM FOREST                                                     #
#                                                                                                             #
###############################################################################################################
library(randomForest)

prop.table(table(Insurance_train$Claimed))
prop.table(table(Insurance_test$Claimed))

RFmodel <- randomForest(formula = Claimed ~ .,data = Insurance_train,ntree = 501,mtry = 4,nodesize = 10,importance = TRUE)

RFmodel$err.rate
plot(RFmodel, main = "Random Forest Model")

randomForestExplainer::plot_importance_rankings(RFmodel)

print(RFmodel)
summary(RFmodel)

importance(RFmodel)

randomForest::varImpPlot(RFmodel,sort = TRUE)

#### ------------------------------- Tuning the Random forest----------------------------------------------####

set.seed(100)
tunedRFmodel <- tuneRF(x = Insurance_train[,-4],y = Insurance_train$Claimed,mtryStart = 4,ntreeTry = 401,stepFactor = 0.5,
                       improve = 0.0001,trace = TRUE,plot = TRUE,doBest = TRUE,nodesize=10,importance=FALSE)

plot(tunedRFmodel)

plot(randomForest::getTree(tunedRFmodel))
randomForest::varImpPlot(tunedRFmodel,sort = TRUE)

randomForestExplainer::plot_importance_rankings(tunedRFmodel)
randomForestExplainer::plot_multi_way_importance(tunedRFmodel)
randomForestExplainer::plot_min_depth_interactions(tunedRFmodel)

###############################################################################################################
#                                      Predict                                                               #
###############################################################################################################

Insurance_trainRF <- Insurance_train
Insurance_testRF <- Insurance_test
Insurance_trainRF$predicted.claim <- predict(tunedRFmodel,Insurance_trainRF,type = "class")
Insurance_trainRF$predicted.score <- predict(tunedRFmodel,Insurance_trainRF,type = "prob")[,2]

Insurance_testRF$predicted.claim <- predict(tunedRFmodel,Insurance_testRF,type = "class")
Insurance_testRF$predicted.score <- predict(tunedRFmodel,Insurance_testRF,type = "prob")[,2]


###############################################################################################################
#                          Random Forest Model Evaluation                                                     #
###############################################################################################################

caret::confusionMatrix(Insurance_trainRF$predicted.claim,Insurance_train$Claimed,positive = "Yes",mode = "everything")

caret::confusionMatrix(Insurance_testRF$predicted.claim,Insurance_test$Claimed,positive = "Yes",mode = "everything")

library(ROCR)

ROCRF<-ROCit::rocit(score = Insurance_trainRF$predicted.score,class = Insurance_trainRF$Claimed)
plot(ROCRF)

predRF <- ROCR::prediction(Insurance_trainRF$predicted.score,Insurance_trainRF$Claimed)
perfRF <- performance(predRF, "tpr", "fpr")

ROCR::plot(perfRF,main = paste("AUC =", 0.911),colorize = TRUE,lwd = 2)
abline(0,1)
RFauc <- performance(predRF,"auc")

ROCRF1<-ROCit::rocit(score = Insurance_testRF$predicted.score,class = Insurance_testRF$Claimed)
plot(ROCRF1)


predRF1 <- ROCR::prediction(Insurance_testRF$predicted.score,Insurance_testRF$Claimed)
perfRF1 <- performance(predRF1, "tpr", "fpr")

ROCR::plot(perfRF1,main = paste("AUC =", 0.801),colorize = TRUE,lwd = 2)
abline(0,1)
RFauc1 <- performance(predRF1,"auc")

###############################################################################################################
#                                                                                                             #
#                           Second part :  Artificial Nueral Network ANN                                      #
#                                                                                                             #
###############################################################################################################

###-----------------------------Data preparation converting to numeric values---------------------------------####

## Reading Data & removing the extreme outlier values and Channel column-------------------------------#######


insurance <- read.csv(file.choose(),header = TRUE)

maxm <- max(insurance$Duration)
minm <- min(insurance$Duration)
insurance <- insurance[!insurance$Duration==maxm,]
insurance <- insurance[!insurance$Duration==minm,]

insurance$Channel <- NULL

### --------------------------------- Loading required Libraries-------------------------------------------##########

library(dummies)
library(dplyr)
library(batman)

### -------------------------- Converting the categorical variables to numeric-------------------------#####

insurance_claimrmvd <- insurance[,-4]

names(insurance_claimrmvd)[names(insurance_claimrmvd) == "Agency_Code"] <- "Agency.Code"

Dummy_insuredata <- dummy.data.frame(insurance_claimrmvd,all = TRUE)

insuredata <- scale(Dummy_insuredata,center = TRUE,scale = TRUE)

insurance$Claimed <- to_logical(insurance$Claimed)

Target_claimd <- insurance %>% dplyr::select(Claimed) %>% dplyr::mutate(Claimed = case_when(Claimed == TRUE ~ 1, FALSE ~ 0))
Target_claimd[is.na(Target_claimd)] = 0

Final_ANNdata <- cbind(insuredata,Target_claimd)

str(Final_ANNdata)
head(Final_ANNdata)

summarytools::view(dfSummary(insurance))
summarytools::view(dfSummary(Final_ANNdata))

### ------------------------------- Removing the spaces from column name-----------------------------------------#####

names(Final_ANNdata)[names(Final_ANNdata) == "TypeTravel Agency"] <- "TypeTravel.Agency"
names(Final_ANNdata)[names(Final_ANNdata) == "Product.NameBronze Plan"] <- "Product.NameBronze.Plan"
names(Final_ANNdata)[names(Final_ANNdata) == "Product.NameCancellation Plan"] <- "Product.NameCancellation.Plan"
names(Final_ANNdata)[names(Final_ANNdata) == "Product.NameCustomised Plan"] <- "Product.NameCustomised.Plan"
names(Final_ANNdata)[names(Final_ANNdata) == "Product.NameGold Plan"] <- "Product.NameGold.Plan"
names(Final_ANNdata)[names(Final_ANNdata) == "Product.NameSilver Plan"] <- "Product.NameSilver.Plan"

### ------------------------------- Split to train and test -----------------------------------------#####

set.seed(123)
datasplitting1 <- initial_split(data = Final_ANNdata,prop = 0.7,strata = "Claimed")

Insurance_trainANN <- training(datasplitting1)
Insurance_testANN <- testing(datasplitting1)


### ------------------------------- Build Nueral Net Model-----------------------------------------#####


library(neuralnet)
library(NeuralNetTools)
library(NeuralSens)


names(Insurance_trainANN)
FormulaforANN <- as.formula(Claimed ~ Age + Agency.CodeC2B + Agency.CodeCWT + Agency.CodeEPX + Agency.CodeJZI + TypeAirlines + TypeTravel.Agency + Commision + Duration + Sales + Product.NameBronze.Plan + Product.NameCancellation.Plan + Product.NameCustomised.Plan + Product.NameGold.Plan + Product.NameSilver.Plan + DestinationAmericas + DestinationASIA + DestinationEUROPE)

set.seed(123)
ANNmodel <- neuralnet(formula =  FormulaforANN,data = Insurance_trainANN,
                      hidden = c(4,2),err.fct = "sse",linear.output = FALSE,
                      lifesign = "full",lifesign.step = 10,threshold = 0.01,stepmax = 150000)

vip::vip(ANNmodel,fill = "dark blue")


### --------- NOTE : Since I gave very low threshold value, the model converges at 112028 iteration -----------------------###
### ----------------- If threshold is at 0.01 or 0.1 the iteration stops at 941 iterations------------------------------####

print(ANNmodel)

ANNmodel$net.result
neuralnet::gwplot(ANNmodel,rep = "best")
plot(ANNmodel,rep = "best")

ANNmodel$result.matrix

NeuralNetTools::plotnet(ANNmodel,bias=TRUE)
NeuralNetTools::olden(mod_in = ANNmodel,x_names)
imp <- NeuralNetTools::olden(mod_in = ANNmodel,bar_plot=FALSE)
NeuralNetTools::neuralweights(ANNmodel)

### -------------------------------------------- Predict -----------------------------------------------------####
set.seed(123)
ANNmodeltest <- neuralnet(formula =  FormulaforANN,data = Insurance_testANN, hidden = c(4,2),err.fct = "sse",linear.output = FALSE,lifesign = "full",lifesign.step = 10,threshold = 0.01,stepmax = 150000)

NeuralNetTools::olden(mod_in = ANNmodel,bar_plot=FALSE)
vip::vip(ANNmodeltest,fill = "dark blue")
NeuralNetTools::olden(mod_in = ANNmodeltest)

# predict_testdata <- compute(ANNmodel,Insurance_testANN)

Insurance_testANN$predicted.class <- predict(ANNmodel,Insurance_testANN,type="raw")
Insurance_testANN$probabilities <- predict(ANNmodel,Insurance_testANN,type="raw")

Insurance_testANN$probabilities <- Insurance_testANN$probabilities[,1]
predict_testdata$net.result

### -------------------------------------------- Model Evaluation -----------------------------------------------------####

Insurance_trainANN$probabilities <- ANNmodel$net.result[[1]]
hist(Insurance_trainANN$probabilities,breaks = 5)
Insurance_trainANN$Claimedpredicted <- ifelse(Insurance_trainANN$probabilities>0.5,1,0)
Insurance_trainANN$Claimedpredicted <- as.factor(Insurance_trainANN$Claimedpredicted)
Insurance_trainANN$Claimed <- as.factor(Insurance_trainANN$Claimed)
Insurance_trainANN$probabilities <- Insurance_trainANN$probabilities[,1]

str(Insurance_testANN)
caret::confusionMatrix(Insurance_trainANN$Claimedpredicted,Insurance_trainANN$Claimed,positive = "1", mode = "everything")

#Insurance_testANN$probabilities <- predict_testdata$net.result[[1]]
hist(Insurance_testANN$probabilities,breaks = 5)


Insurance_testANN$predicted.class <- ifelse(Insurance_testANN$probabilities >0.5,1,0)
Insurance_testANN$predicted.class <- as.factor(Insurance_testANN$predicted.class)
Insurance_testANN$Claimed <- as.factor(Insurance_testANN$Claimed)

caret::confusionMatrix(Insurance_testANN$predicted.class,Insurance_testANN$Claimed,positive = "1", mode = "everything")

#### -----------------------------ROCR -------------------------------------------------------------------######

library(ROCR)

ROCANN<-ROCit::rocit(score = Insurance_trainANN$probabilities,class = Insurance_trainANN$Claimed)
plot(ROCANN)

predANN <- ROCR::prediction(Insurance_trainANN$probabilities,Insurance_trainANN$Claimed)
perfANN <- performance(predANN, "tpr", "fpr")

trannauc <- AUC(Insurance_trainANN$Claimedpredicted,Insurance_trainANN$Claimed)
trannauc <- round(trannauc,digits = 3)

ROCR::plot(perfANN,main = paste("AUC = ", trannauc),colorize = TRUE,lwd = 2)
abline(0,1)

AUC(Insurance_trainANN$Claimedpredicted,Insurance_trainANN$Claimed)

#--------------------Test Data ----------------------------------####
library(ROCit)
ROCANN1<-ROCit::rocit(score = Insurance_testANN$probabilities,class = Insurance_testANN$Claimed)
plot(ROCANN1)

predANN1 <- ROCR::prediction(Insurance_testANN$probabilities,Insurance_testANN$Claimed)
perfANN1 <- performance(predANN1, "tpr", "fpr")

ROCR::plot(perfANN1,main = paste("AUC = ", testnauc),colorize = TRUE,lwd = 2)
abline(0,1)

testnauc<- AUC(Insurance_testANN$predicted.class,Insurance_testANN$Claimed)
testnauc <- round(testnauc,digits = 3)

### ---------------------------- Conclusion Visuals -----------------------------------------------------#####

Claimeddf <- subset(insurance,insurance$Claimed == "Yes")

Claimeddf$Claimed <- NULL

library(esquisse)
esquisse::esquisser()


sales <- cut(insurance$Sales, include.lowest = TRUE, breaks = seq(0, 600, by = 30))
ggplot(insurance, aes(sales, fill = Claimed)) + geom_bar(position="dodge") + scale_fill_manual(values=c("blue","Red")) + geom_text(stat='count',aes(label = after_stat(count)),vjust = -1)

commision <- cut(insurance$Commision, include.lowest = TRUE, breaks = seq(0, 300, by = 10))
ggplot(insurance, aes(commision, fill = Claimed)) + geom_bar(position="dodge") + scale_fill_manual(values=c("blue","Red")) + geom_text(stat='count',aes(label = after_stat(count)),vjust = -1)

duration_of_travel <- cut(insurance$Duration, include.lowest = TRUE, breaks = seq(0, 500, by = 25))
ggplot(insurance, aes(duration_of_travel, fill = Claimed)) + geom_bar(position="dodge") + scale_fill_manual(values=c("blue","Red")) + geom_text(stat='count',aes(label = after_stat(count)),vjust = -1)