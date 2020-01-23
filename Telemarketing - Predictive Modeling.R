remove(list = ls())
library(tidyverse)
library(dplyr)
library(sandwich)
library(knitr)
library(lmtest)
library(car)
library(data.table)
library(glmnet)
library(pastecs)
library(caret)
library(pls)
library(standardize)
library(psych)
library(corrplot)
library(rpart)
library(rpart.plot)
library(caTools)
library(RColorBrewer)
library(rattle)


############## BASIC EXPLORATORY ANALYSIS#####
### Q1)#####


dta_bank<- read.csv(file="/Users/salmamohammed/Downloads/Bank Case.csv")
class(dta_bank)
dta_bank<-as.data.table(dta_bank)


### Q2)##### 

# 2a) Units: age is years, and duration is also years.
list<-colnames(dta_bank)
sapply(dta_bank, class)

#2) b) 
describe(dta_bank)
if (max(dta_bank$age)>(mean(dta_bank$age)+4*(sd(dta_bank$age)))){print("has an outlier")}
if (max(dta_bank$duration)>(mean(dta_bank$duration)+4*(sd(dta_bank$duration)))){print("has an outlier")}
if (min(dta_bank$age)<(mean(dta_bank$age)-4*(sd(dta_bank$age)))){print("has an outlier")}
if (min(dta_bank$duration)<(mean(dta_bank$duration)-4*(sd(dta_bank$duration)))){print("has an outlier")}

### Q3)#####


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


corrs<-cor(dta_bank_d)
corrplots<-corrplot(corrs)
### Q4)##### 

#a) y= 0.028+ 0.000798jobblue-collar -.02669jobentrepreneur -.004315jobhousemaid...+
#        .007maritalmarried+ .02247maritalsingle +...+ .005educationbasic.6y+ ...-.0388defaultunknown +
#       .0009568housingunknown+... -.0039loanyes -.08contacttelephone+... + .2327monthdec+ -.01007day_of_weekmon      
#
#   b) i. Best time to perform telemarketing tasks is Wednesday in March
#      ii. Retired Groups are more likely to respond.
#      iii. Health could be a source of omitted variable bias. I wonder if people with 
#           health issues would be more likely to engage with the bank due to financial pressures.

lm1<-lm(y~., data=dta_bank)
summary(lm1)





############## PREDICTOVE MODELING AND TUNING ####

### Q1)####
# We split it to 80% training, 10% testing, and 10% validating. 
# We do that so that we can train our model on a set of data, then we get to 
# test our model on another set of data (validating data). And we repeat the process
# Once we find a good model, we can finally use it for a final test to predict
# the values of a test data set (test data). The reason we take all these precausions 
# is so that our model doesn't start predicting the error term and gets too tailored to our 
# one data set. The model should help us predict other data, that it has not been trained on. 
# if it fails to do that, then it is a useless model.

### Q2)####
# Yes, call duration since it happens after the call (it's an outcome) 
# rather than a background information that we can use to predict the 
# outcome of the call. This is also indicated as part of the description of the data.

### Q3)#### 
# Overfitting is when our validating model has more accuracy than our test model.
# According to google: it is a condition where a statistical model begins to 
# describe the random error in the data rather than the relationships between variables.
# Underfitting is the opposite and is not much of a problem.

### Q4)#### 
# This just means that some models work better with certain datasets.
# In other words: some algorithms will fit certain data sets better than others. 
# It also (by definition) means that there will be as many data sets that a given 
# algorithm will not be able to model effectively. How effective a model will be is 
# directly dependent on how well the assumptions made by the model fit the true nature of the data. (source: www.kdnuggets.com)

## Prepping the data: 

# Drop duration:
dta_bank <- dta_bank %>% mutate(duration=NULL)
dta_bank_d <- dta_bank_d %>% mutate(duration=NULL)


# Split the data:
set.seed(1890)
inx_train    = caret::createDataPartition(dta_bank$y, p=0.8)$Resample1 
dta_train    = dta_bank[ inx_train, ]
dta_left     = dta_bank[-inx_train, ]
inx_test     = caret::createDataPartition(dta_left$y, p=0.5)$Resample1
dta_test     = dta_left[ inx_test, ]
dta_valid    = dta_left[ -inx_test, ]

unique(dta_bank_d$y)
unique(dta_bank$y)


### Q5)####
# a) They one that overfits the most is LM4.
# b) None of them underfits. The one that overfits the least is LM2.
# c) Not necessarily. If the model fits the training data the best, but underperforms
# in predicting the test data then that is an issue.
# d) Yes, a confusion matrix can help us calculate the accuracy of our model for both the training and test datasets.
# e) To run these regressions, we will first train the model on the training data, then we can validate our model
# by using it to predict the validating data. We can repeat this process until we feel happy with our model. Finally, we 
# should test our model on the test data by predicting its y. 
# LM1####
# Model structure: y=0.2 - 0.000018age - 0.1month(aug)+ 0.2month(dec)...
lm1<-lm(y~ age + factor(month), data= dta_train)
summary(lm1) # AR2 0.07459 
pr_lm1.v<- predict(lm1, newdata=dta_valid)
pr_lm1.v <- as.data.table(pr_lm1.v)
pr_lm1.v<- pr_lm1.v %>% mutate(pr_lm1.v = ifelse(pr_lm1.v>=.5, 1,0))


fitted_data          = data.table( cbind(real_data = as.numeric(dta_valid$y),
                                         pred_data = as.numeric(pr_lm1.v$pr_lm1.v))
)

fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
dim(fitted_data)
confuss_mat_prog
accuracy<-correct/all
accuracy
#Valid Data Model Accuracy is 0.8960661


pr_lm1.t<- predict(lm1, newdata=dta_test)
pr_lm1.t <- as.data.table(pr_lm1.t)
pr_lm1.t<- pr_lm1.t %>% mutate(pr_lm1.t = ifelse(pr_lm1.t>=.5, 1,0))

fitted_data          = data.table( cbind(real_data = as.numeric(dta_test$y),
                                         pred_data = as.numeric(pr_lm1.t$pr_lm1.t))
)

fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
dim(fitted_data)
confuss_mat_prog
accuracy<-correct/all
accuracy
#Test Data Model Accuracy is 0.8917213



# LM2####
# Model Structure: y= 0.8073- 0.03.3age + .000505age_sqrd+ ...+ 0.27month(sep)
dta_train_lm2 <- dta_train
dta_train_lm2 <- dta_train_lm2 %>% mutate(age_sqrd=age*age,
                                          age_cube=age*age*age)

dta_valid_lm2 <- dta_valid
dta_valid_lm2 <- dta_valid_lm2 %>% mutate(age_sqrd=age*age,
                                          age_cube=age*age*age)

dta_test_lm2 <- dta_test
dta_test_lm2 <- dta_test_lm2 %>% mutate(age_sqrd=age*age,
                                        age_cube=age*age*age)
  
lm2<-lm(y~age+age_sqrd+age_cube+factor(month),data= dta_train_lm2)
summary(lm2) # AR2 0.08796
pr_lm2.v<- predict(lm2, newdata=dta_valid_lm2)
pr_lm2.v <- as.data.table(pr_lm2.v)
pr_lm2.v<- pr_lm2.v %>% mutate(pr_lm2.v = ifelse(pr_lm2.v>=.5, 1,0))


fitted_data          = data.table( cbind(real_data = as.numeric(dta_valid_lm2$y),
                                         pred_data = as.numeric(pr_lm2.v$pr_lm2.v))
)

fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
dim(fitted_data)
confuss_mat_prog
accuracy<-correct/all
accuracy
#Valid Data Model Accuracy is 0.8933949


pr_lm2.t<- predict(lm2, newdata=dta_test_lm2)
pr_lm2.t <- as.data.table(pr_lm2.t)
pr_lm2.t<- pr_lm2.t %>% mutate(pr_lm2.t = ifelse(pr_lm2.t>=.5, 1,0))

fitted_data          = data.table( cbind(real_data = as.numeric(dta_test_lm2$y),
                                         pred_data = as.numeric(pr_lm2.t$pr_lm2.t))
)

fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
dim(fitted_data)
confuss_mat_prog
accuracy<-correct/all
accuracy
#Test Data Model Accuracy is 0.890993


# LM3#### 
# Model Structure:   y= 0.028+ 0.000798jobblue-collar -.02669jobentrepreneur -.004315jobhousemaid...+
#                       .007maritalmarried+ .02247maritalsingle +...+ .005educationbasic.6y+ ...-.0388defaultunknown +
#                       .0009568housingunknown+... -.0039loanyes -.08contacttelephone+... + .2327monthdec+ -.01007day_of_weekmon 
lm3 = lm(y~., data= dta_train)
summary(lm3) # AR2 0.1021 
pr_lm3.v<- predict(lm3, newdata=dta_valid)
pr_lm3.v <- as.data.table(pr_lm3.v)
pr_lm3.v<- pr_lm3.v %>% mutate(pr_lm3.v = ifelse(pr_lm3.v>=.5, 1,0))


fitted_data          = data.table( cbind(real_data = as.numeric(dta_valid$y),
                                         pred_data = as.numeric(pr_lm3.v$pr_lm3.v))
)

fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
dim(fitted_data)
confuss_mat_prog
accuracy<-correct/all
accuracy
#Valid Data Model Accuracy is 0.8984944


pr_lm3.t<- predict(lm3, newdata=dta_test)
pr_lm3.t <- as.data.table(pr_lm3.t)
pr_lm3.t<- pr_lm3.t %>% mutate(pr_lm3.t = ifelse(pr_lm3.t>=.5, 1,0))

fitted_data          = data.table( cbind(real_data = as.numeric(dta_test$y),
                                         pred_data = as.numeric(pr_lm3.t$pr_lm3.t))
)

fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
dim(fitted_data)
confuss_mat_prog
accuracy<-correct/all
accuracy
#Test Data Model Accuracy is 0.8922068


# LM4####


lm4 = lm(y~.^2, data= dta_train)
summary(lm4) # AR2 0.1469  
pr_lm4.v<- predict(lm4, newdata=dta_valid)
pr_lm4.v <- as.data.table(pr_lm4.v)
pr_lm4.v<- pr_lm4.v %>% mutate(pr_lm4.v = ifelse(pr_lm4.v>=.5, 1,0))


fitted_data          = data.table( cbind(real_data = as.numeric(dta_valid$y),
                                         pred_data = as.numeric(pr_lm4.v$pr_lm4.v))
)

fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
dim(fitted_data)
confuss_mat_prog
accuracy<-correct/all
accuracy
#Valid Data Model Accuracy is 0.8950947


pr_lm4.t<- predict(lm4, newdata=dta_test)
pr_lm4.t <- as.data.table(pr_lm4.t)
pr_lm4.t<- pr_lm4.t %>% mutate(pr_lm4.t = ifelse(pr_lm4.t>=.5, 1,0))

fitted_data          = data.table( cbind(real_data = as.numeric(dta_test$y),
                                         pred_data = as.numeric(pr_lm4.t$pr_lm4.t))
)

fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
dim(fitted_data)
confuss_mat_prog
accuracy<-correct/all
accuracy
#Test Data Model Accuracy is 0.8885652

############## IMPROVING THE PREDICTIVE POWER#### 
### Q1)####

# Changing the y back to a scale between 0 to 1 (instead of a categorical)
pr_lm2.v<- predict(lm2, newdata=dta_valid_lm2)
pr_lm2.v <- as.data.table(pr_lm2.v)
summary(pr_lm2.v)

pairs(~ y + age + age_sqrd +age_cube + month, 
      data = dta_valid_lm2, row1attop=FALSE)

# there is no apparent linear relationship between y and the other 
# variables (age, age^2 and age^3 have a linear relationship between them as expected). 
### Q2)####
# NAIVE BAYES MODEL####

set.seed(1890)
unique(dta_bank$y)
inx_train    = caret::createDataPartition(dta_bank$y, p=0.8)$Resample1 
dta_train    = dta_bank[ inx_train, ]
dta_left     = dta_bank[-inx_train, ]
inx_test     = caret::createDataPartition(dta_left$y, p=0.5)$Resample1
dta_test     = dta_left[ inx_test, ]
dta_valid    = dta_left[ -inx_test, ]
dta_train$y<-as.factor(dta_train$y)

dta_train_lm2 <- dta_train
dta_train_lm2 <- dta_train_lm2 %>% mutate(age_sqrd=age*age,
                                          age_cube=age*age*age)

dta_valid_lm2 <- dta_valid
dta_valid_lm2 <- dta_valid_lm2 %>% mutate(age_sqrd=age*age,
                                          age_cube=age*age*age)

dta_test_lm2 <- dta_test
dta_test_lm2 <- dta_test_lm2 %>% mutate(age_sqrd=age*age,
                                        age_cube=age*age*age)

NBclassifier= naivebayes::naive_bayes(formula= y~age+ age_sqrd+age_cube+factor(month),
                                     usekernel = T,
                                     data      = dta_train_lm2)
predict(NBclassifier,newdata = dta_train_lm2)

fitted_data$real_data

# Evaluating model performance using a confusion matrix            
fitted_data          = data.table( cbind(real_data = dta_valid_lm2$y,
                                         pred_data = paste(predict(NBclassifier,newdata = dta_valid_lm2)))
)

class(fitted_data$real_data)
class(fitted_data$pred_data)
fitted_data<-as.data.table(fitted_data)
fitted_data$is_equal = fitted_data$real_data==fitted_data$pred_data
confuss_mat_prog          = fitted_data[,
                                        {
                                          tmp1=sum(is_equal);
                                          tmp2=sum(!is_equal);
                                          list(corrects=tmp1,wrongs = tmp2)
                                        },keyby=.(real_data,pred_data)]
all<-    sum(confuss_mat_prog[,3:4])
correct<-sum(confuss_mat_prog[,3])
confuss_mat_prog


accuracy<-correct/all

accuracy

# Accuracy for the NB model is  0.8926663. 
# The model doesn't give us a better accuracy than the linear model.


# KNN MODEL####

#factoring age & y
#dta_train_kn<- dta_train
#dta_train_kn$y <- as.factor((dta_train_kn$y))


dta_bank <- dta_bank %>% mutate(duration=NULL)
dta_bank_d <- dta_bank_d %>% mutate(duration=NULL)
set.seed(1890)
inx_train    = caret::createDataPartition(dta_bank_d$y, p=0.8)$Resample1 
dta_train    = dta_bank_d[ inx_train, ]
dta_left     = dta_bank_d[-inx_train, ]
inx_test     = caret::createDataPartition(dta_left$y, p=0.5)$Resample1
dta_test     = dta_left[ inx_test, ]
dta_valid    = dta_left[ -inx_test, ]


dta_train_lm2 <- dta_train
dta_train_lm2 <- dta_train_lm2 %>% mutate(age_sqrd=age*age,
                                          age_cube=age*age*age)

dta_valid_lm2 <- dta_valid
dta_valid_lm2 <- dta_valid_lm2 %>% mutate(age_sqrd=age*age,
                                          age_cube=age*age*age)

dta_test_lm2 <- dta_test
dta_test_lm2 <- dta_test_lm2 %>% mutate(age_sqrd=age*age,
                                        age_cube=age*age*age)


dim(dta_train_lm2)
dim(dta_test_lm2)



# Normalizing function and normalizing age                                                
normalize = function(x){return ((x - min(x)) / (max(x) - min(x)))}
normalize(c(1, 2, 3, 4, 5))
normalize(c(10, 20, 30, 40, 50))
dta_train_lm2$age = normalize(dta_train_lm2$age)
dta_train_lm2$age = normalize(dta_train_lm2$age_sqrd)
dta_train_lm2$age = normalize(dta_train_lm2$age_cube)

dta_valid_lm2$age = normalize(dta_valid_lm2$age)
dta_valid_lm2$age = normalize(dta_valid_lm2$age_sqrd)
dta_valid_lm2$age = normalize(dta_valid_lm2$age_cube)

dta_test_lm2$age = normalize(dta_test_lm2$age)
dta_test_lm2$age = normalize(dta_test_lm2$age_sqrd)
dta_test_lm2$age = normalize(dta_test_lm2$age_cube)
summary(dta_train_lm2$age)

# Training model on dta_training 
require("class")
knn_model <-  knn(dta_train_lm2, dta_test_lm2, dta_train_lm2$y, k=3)

dim(dta_test)
# Evaluating performance on dta_test  
k1_conf_mat  =   gmodels::CrossTable(x          = dta_test_lm2$y, 
                                     y          = knn_model,
                                     prop.chisq = TRUE)

# Confusion matrix
k1_conf_mat$t
accuracy <- (3655+164)/(4119)
accuracy

# ACCURACY OF KNN MODEL IS 0.9271668!!! 

### Q3)####
# 3) NB makes the accuracy worse, whereas KNN gives us the best accuracy!!
############### CAUSAL QUESTIONS ####
### Q1)####
# 1) When we study causality we always focus on the parameters multiplying the X variables
#instead of the predictive capacity of the model. We then give a causal interpretation to
#the estimated coefficients.
#a. Explain when in marketing is preferable a causal analysis to a predictive analysis.
#ANS: A causal analysis is preferable to predictive analysis when we are trying to understand what causes 
# a certain reaction from our customers.
#b. In the context of a linear regression, explain the concepts of a biased estimated.
#Ans: A biased estimate is when our our actual Betas are far from our estimated Betas in one direction vs. the other (instead of just having a larger variance). 
# According to Wikipidea, bias is related to consistency in that consistent estimators are convergent and asymptotically unbiased (hence converge to the correct 
# value as the number of data points grows arbitrarily large).

### Q2)####
# 2 . Which of the variables could be interesting to analyze from a causal point of view. Give
#examples. 
# ANS: It would be interesting to see if marriage (or perhaps even having kids) causes people to engage with the bank.



### Q3)####

# 3. For those variables what would be the potential omitted variables problem?
# ANS: Marriage is related to soci-economic status to some extent (most of married people are ones who can afford it), 
# it can also be tied to religion (if interacted with age) and religion can impact people's banking habits,
# so this could be causing omitted variable bias.
# The same applies to number of kids. Rich people tend to have fewer kids, 
# where as larger families tend to be poor. This could be correlated with the error term.








