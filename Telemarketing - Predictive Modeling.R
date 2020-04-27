# Salma Mohammed, 2020 (Updated)

#### PROBLEM AT HAND ####
# The problem we are trying to solve is coming up with the best classification predictive model that can help us 
# predict the outcome (answers call vs. doesnt answer) of this bank's tele-marketing phone calls.

#### IMPORTING LIBRARIES AND DATASET ####
remove(list = ls())
library(tidyverse)
library(dplyr)
library(knitr)
library(car)
library(data.table)
library(pastecs)
library(caret)
library(psych)
library(corrplot)
library(rpart)
library(rpart.plot)
library(rattle)
library(ggplot2)
library(ROSE)
library(pROC)
library(class)
library(randomForest)



# Importing data set
dta_bank<- as.data.table(read.csv(file="Bank Case.csv"))
  
#### EXAMINING THE DATA ####



# Examining Data Types
glimpse(dta_bank)
list<-colnames(dta_bank)
sapply(dta_bank, class)

describe(dta_bank)

  

#### DATA PREPERATION I ####
 

# Examining Data Types
glimpse(dta_bank)
list<-colnames(dta_bank)
sapply(dta_bank, class)

# Removing outliers
if (max(dta_bank$age)>(mean(dta_bank$age)+4*(sd(dta_bank$age)))){print("has an outlier")}
if (max(dta_bank$duration)>(mean(dta_bank$duration)+4*(sd(dta_bank$duration)))){print("has an outlier")}
if (min(dta_bank$age)<(mean(dta_bank$age)-4*(sd(dta_bank$age)))){print("has an outlier")}
if (min(dta_bank$duration)<(mean(dta_bank$duration)-4*(sd(dta_bank$duration)))){print("has an outlier")}


  


#### DATA PREPERATION II ####

 
# Creating Dummies
dta_bank<- dta_bank %>% mutate(y=ifelse(y=='yes', 1,0))
dta_bank_d <- fastDummies::dummy_cols(dta_bank, remove_first_dummy = TRUE)
dta_bank_d <- dta_bank_d %>% mutate(job=NULL,
                                    marital=NULL,
                                    education=NULL,
                                    default=NULL,
                                    housing=NULL,
                                    loan=NULL,
                                    contact=NULL,
                                    month=NULL,
                                    day_of_week=NULL
)


colnames(dta_bank_d)

# Summary statistics and missingness
describe(dta_bank_d)
unique(is.na(dta_bank_d)) # no missingness

# Correlation 
corrs<-round(cor(dta_bank_d),2)  
corrs
dta_bank_d$duration <-NULL
#the table shows intermediate correlation between y and duration which makes sense since there would only be a duration for the call if the call was picked up in the first place, hence, we can drop that variable from our data set. 
# There was high correlation between married vs single, as well as age & retirement, so we will need to examine that more through looking at the VIF of those two variables once we run our initial regression. Finally, there was some correlation between educational and job variables. 



# Examining the balance of our data: the distribution of categorical outcome variable (y) 
ggplot(dta_bank_d, aes(x=factor(y, levels=c(1, 0)))) + 
  geom_bar(stat = 'count', fill='steelblue2', width = 0.6) + 
  labs(title = '', 
       x     = 'Answered Call',
       y     = 'Count') +
  scale_x_discrete(drop=FALSE, labels = c('Yes','No')) + theme_minimal()

# our data is very imbalanced, so we will need to upsample our under-represented group (Answered-calls)


# Adjusting our data imbalance using SMOTE
print(prop.table(table(dta_bank_d$y)))
colnames(dta_bank_d)

# Renaming some variables
names(dta_bank_d)[names(dta_bank_d)=="job_blue-collar"] <- "job_blue_collar"
names(dta_bank_d)[names(dta_bank_d)=="job_self-employed"] <- "job_self_employed"


# Spliting the data
set.seed(1890)
inx_train    = createDataPartition(dta_bank_d$y, p=0.8)$Resample1 
dta_train    = dta_bank_d[ inx_train, ]
dta_left     = dta_bank_d[-inx_train, ]
inx_test     = createDataPartition(dta_left$y, p=0.5)$Resample1
dta_test     = dta_left[ inx_test, ]
dta_valid    = dta_left[ -inx_test, ]

# Checking the distribution of outcome variable in training and test data. They are both distributed similarly.
print(prop.table(table(dta_train$y)))
print(prop.table(table(dta_test$y)))


# Generating synthetic data for our training sample
dta_train = ROSE(y ~ ., data = dta_train, seed = 1)$data


# Checking the distribution of new dataset
print(prop.table(table(dta_train$y))) # We now have a balanced distribution for our training data
ggplot(dta_train, aes(x=factor(y, levels=c(1, 0)))) + 
  geom_bar(stat = 'count', fill='steelblue2', width = 0.6) + 
  labs(title = '', 
       x     = 'Answered Call',
       y     = 'Count') +
  scale_x_discrete(drop=FALSE, labels = c('Yes','No')) + theme_minimal()
  


#### LOGISTIC REGRESSION & searching for large variance inflation factors (VIF) ####
 
summary(dta_train)
train_logit<- glm(y ~ ., data = dta_train, family = "binomial"(link = "logit"))
summary(train_logit)

# Checking VIF
alias( lm( y ~ . , data=dta_train) )
dta_train$housing_unknown <- NULL
round(vif(train_logit),1)
# The variable with the highest VIF is  education_university.degree but it remains within an acceptable range (below 5) so we will keep the variable in the model. 


# Running a regression with only variables with statistic significance (p<0.05)
train_logit2<-glm(as.formula(paste(colnames(dta_train)[2], "~",
                                   paste(colnames(dta_train)[-c(5:6, 13, 14, 16, 17, 19:21, 23, 25:28 )], collapse = "+"), sep = "")),
                  data=dta_train, family = "binomial"(link = "logit"))
summary(train_logit2)


# Using our model to predict on our validation data
logit_pred_2 <- predict(train_logit2, dta_valid, type = "response")
vif(train_logit2)

# Checking first 5 actual and predicted records
data.frame(actual = dta_valid$y[1:5], predicted = logit_pred_2[1:5]) # outcome is looking great so far! all were predicted correctly


# Confusion Matrix 
logit_pred_2 <-round(logit_pred_2,5)
logit_pred_2 <- as.factor(ifelse(logit_pred_2 > 0.5, 1, 0))
confusionMatrix(logit_pred_2 , as.factor(dta_valid$y)) #Accuracy : 0.6804, Sensitivity : 0.6808, Specificity : 0.6776


plot(roc(logit_pred_2, dta_valid$y, direction="<"),
     col="Blue", lwd=3, main="ROC")

# Conductin Outlier Analysis - The Normal Q-Q plot looks very off which is problematic. We should explore other models to see if they perform significantly better than the logistic model, or come back to this section and expirement with removing certain observations that would help with fixing the Normal Q-Q graph. 
hist(train_logit2$residuals)
par(mfrow=c(2,2))
plot(train_logit2)


  

#### K NEAREST NEIGHBOR ####
 

# Partitioning the data:
set.seed(1890)
inx_train    = createDataPartition(dta_bank_d$y, p=0.8)$Resample1 
dta_train    = dta_bank_d[ inx_train, ]
dta_left     = dta_bank_d[-inx_train, ]
inx_test     = createDataPartition(dta_left$y, p=0.5)$Resample1
dta_test     = dta_left[ inx_test, ]
dta_valid    = dta_left[ -inx_test, ]

# Normalizing the data (the age variable since it's the only continuous one)
normalize = function(x){return ((x - min(x)) / (max(x) - min(x)))}

dta_train_knn <- dta_train
dta_test_knn  <- dta_test
dta_valid_knn <- dta_valid

dta_train_knn[,1] <- normalize(dta_train[,1])
dta_test_knn[,1]  <- normalize(dta_test[,1])
dta_valid_knn[,1] <- normalize(dta_valid[,1])

# Removing our outcome variable
dta_train_knn_X = dta_train_knn[,-2]
dta_test_knn_X  = dta_test_knn[,-2]
dta_valid_knn_X = dta_valid_knn[,-2]

# Creating a data frame with two columns: k, and accuracy
accuracy.df <- data.frame(k = seq(1, 12, 1), accuracy = rep(0, 12))

# Using our model to predict on our validating data and computing knn for different k on validation to find the optimal k value
set.seed(1890)
for(i in 2:12) {          
  knn.pred <- knn(dta_train_knn_X, dta_valid_knn_X, dta_train_knn$y, k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, as.factor(dta_valid_knn$y))$overall[1] 
}

accuracy.df # K=9 is our optimal value for K as it acheives the highest level of accuracy




  


#### DECISION TREE MODEL ####
 

# Creating an unpruned tree
tree_model_unpruned <- rpart(y ~ ., data = dta_train)
fancyRpartPlot(tree_model_unpruned, type=2, caption="", palettes=c("PuBu", "OrRd"), tweak=1) 

# Using our model to predict on our validating data
valid_pred_tree = predict(tree_model_unpruned, newdata = dta_valid)
valid_pred_tree <-round(valid_pred_tree, 5)
valid_pred_tree <- as.factor(ifelse(valid_pred_tree > 0.5, 1, 0))

confusionMatrix(valid_pred_tree, as.factor(dta_valid$y)) #Accuracy:0.8975, Sensitivity:0.9889 ,Specificity: 0.1098


# Pruning our decision tree
train_for_cv = rbind(dta_train, dta_valid)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(123)
tree_model_pruned <- train(factor(y) ~., 
                           data = train_for_cv, 
                           method = "rpart",
                           trControl=trctrl,
                           tuneLength = 15)

plot(tree_model_pruned) 
tree_model_pruned # best complexity parameter: 0.0009537434  Accuracy = 0.8879029 which is lower than our unpruned tree so we can leave our tree unpruned. 


  


#### RANDOM FOREST ####
 

# Runing a random forest model
set.seed(123)
rf_model = randomForest(as.factor(y) ~ ., 
                        data = dta_train, 
                        ntree = 500, 
                        mtry = 3, 
                        importance = TRUE)  

# Using our model to predict on our validating data
rf_pred = predict(rf_model, dta_valid)
confusionMatrix(rf_pred, as.factor(dta_valid$y)) #Accuracy: 0.8968, Sensitivity: 0.99919, Specificity : 0.01402

# Variable importance plot 
varImpPlot(rf_model, type = 1)
# It seems that age has a large impact as a variable on the outcome variable (y) followed by what month the person was contacted, and whether they were contacted by telephone.

# Optimizing and tuning our model
trctrl <- trainControl(method="repeatedcv", number=10, repeats = 3)

rf_model_cv <- train(
  as.factor(y) ~ .,
  tuneLength = 3,
  data = train_for_cv, 
  method = 'ranger',
  trControl = trctrl
)

rf_model_cv$finalModel
rf_model_cv$bestTune
# our best model uses 2 variables at each split and and the minimum size of nodes is 1

  

#### TESTING MODELS #####


# LOGISTIC:  <- this will probably perform the worst based on the residual analysis we ran
logit_pred_test <- predict(train_logit2, dta_test, type = "response")

# LOGISTIC: Confusion Matrix 
logit_pred_test <-round(logit_pred_test,5)
logit_pred_test <- as.factor(ifelse(logit_pred_test > 0.5, 1, 0))
print(confusionMatrix(logit_pred_test , as.factor(dta_test$y))$overall[1]) # Accuracy: 0.6836611 



# KNN: 
knn.pred.test <-  knn(dta_train_knn_X, dta_test_knn_X, dta_train_knn$y, k=9)

# KNN: Confusion Matrix
print(confusionMatrix(knn.pred.test, as.factor(dta_test_knn$y))$overall[1]) # Accuracy: 0.8839524 



# DECISION TREE:  <- our unpruned tree performed better than the pruned one so we will be using this model
test_pred_tree = predict(tree_model_unpruned, newdata = dta_test)
test_pred_tree <-round(test_pred_tree, 5)
test_pred_tree <- as.factor(ifelse(test_pred_tree > 0.5, 1, 0)) 

# DECISION TREE: Confusion Matrix 
print(confusionMatrix(test_pred_tree, as.factor(dta_test$y))$overall[1]) # Accuracy: 0.890993 


# RANDOM FOREST: 
rf_pred = predict(rf_model, dta_test)


# RANDOM FOREST: Confusion Matrix:
print(confusionMatrix(rf_pred, as.factor(dta_test$y))$overall[1]) #Accuracy: 0.8917213



#### CONCLUSION ####
#Based on Accuracy scores, the best predictive model to use is the Random Forest model (0.8917213). 
#In this excercise we didn't focus on Sensitivity and Specifity since in the context of the probelm they are both important: We
#wouldn't want to predict that a customer, who did in fact pick up our call, wasn't going to pick it up because that way we might 
#lose out on an opportunity with that customer. Similarly, we wouldn't want to waste our resources reaching out to customers who we had #predicted were going to pick up our calls, but end up not doing so. So, for our purposes, we relied on Accuracy.





