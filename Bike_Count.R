#Removing all the object stored
rm(list = ls())

#Setting the working directory to the desired location
setwd('D:/Data Science/Edwisor/Project')

#Loading required Libraries

library_vec= c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "dummies", "e1071", "Information",
               "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(library_vec, require, character.only= TRUE)

#Loading the CSV file in R data frame
bike_data= read.csv('day.csv', header = T, na.strings = c("", " ","NA"))

#Visualization
# Analysing the bike count based on season

ggplot(bike_data, aes_string(x= bike_data$season, y= bike_data$cnt)) +
  geom_bar(stat = "identity", fill= "DarkslateBlue") + theme_bw() + 
  xlab("Season") + ylab("Total no. of bike count") + scale_y_continuous() +
  ggtitle("Analysis of bike count Based on Season") + theme(text = element_text(size = 15))

#Analysing the bike count based on Weather Condition
ggplot(bike_data, aes_string(x= bike_data$weathersit, y= bike_data$cnt)) +
  geom_bar(stat = "identity", fill= "Red") + theme_bw() + 
  xlab("Weather Situation") + ylab("Total no. of bike count")  + scale_y_continuous()
  ggtitle("Analysis of bike count Based on Weather Situation") + theme(text = element_text(size = 15))

#Looking at the structure of the data to get a brief idea
str(bike_data)
#Removing predictor variable "instant" from the data as it is storing the index value only.
bike_data= subset(bike_data, select= -instant)
#Exploratory data Analysis
bike_data$season= as.factor(bike_data$season)
bike_data$yr= as.factor(bike_data$yr)
bike_data$mnth= as.factor(bike_data$mnth)
bike_data$holiday= as.factor(bike_data$holiday)
bike_data$weekday= as.factor(bike_data$weekday)
bike_data$workingday= as.factor(bike_data$workingday)
bike_data$weathersit= as.factor(bike_data$weathersit)

d1= unique(bike_data$dteday)
df= data.frame(d1)
# In our data frame we have dteday variable which tells the date in format of yy-mm-dd, also we
#have variables like mnth and yr which gives info about the month and year respectively.
#So converting the the yy-mm-dd to day only
bike_data$dteday=as.Date(df$d1,format="%Y-%m-%d")
df$d1=as.Date(df$d1,format="%Y-%m-%d")
bike_data$dteday=format(as.Date(df$d1,format="%Y-%m-%d"), "%d")
bike_data$dteday=as.factor(bike_data$dteday)

################ Missing Value Analysis ####################
# Creating a dataframe which will tell the amount of Missing value in each variable

missing_val= data.frame(apply(bike_data, 2, function(x){sum(is.na(x))}))

################## Outlier Analysis ########################
# Here we will use Boxplot method to check the prsence of outlier in numeric variable as we
# can apply outlier analysis only on numeric variable.

#Selecting only numeric variable
numeric_index= sapply(bike_data, is.numeric)
numeric_data= bike_data[, numeric_index]

cnames= colnames(numeric_data)

for (i in 1: length(cnames)){
  assign(paste0("gn", i), ggplot(aes_string(y= (cnames[i]), x= "cnt"), data = subset(bike_data)) +
           stat_boxplot(geom = "errorbar", width= 0.5) +
           geom_boxplot(outlier.color = "red", fill= "grey", outlier.shape = 18,
                        outlier.size = 1, notch = FALSE) +
           theme(legend.position = "bottom") +
           labs(y=cnames[i], x= "cnt") +
           ggtitle(paste("Boxplot of Count of rental bikes for", cnames[i])))
}


gridExtra::grid.arrange(gn1, gn2, ncol= 2)
gridExtra::grid.arrange(gn3, gn4, ncol= 2)
gridExtra::grid.arrange(gn5, gn6, ncol= 2)

# Replacing all the outliers with NA so that we can treat it like a missing value and impute it.
for(i in cnames){
  val= bike_data[, i][bike_data[, i] %in% boxplot.stats(bike_data[, i])$out]
  bike_data[, i][bike_data[, i] %in% val]= NA
}

#Now imputing the missing value using KNN- Imputation
bike_data= knnImputation(bike_data, k=5)

################ Feature Selection ##################
# Correlation Plot/ Correlational Analysis
corrgram(bike_data[, numeric_index], order = F, upper.panel = panel.pie,
         text.panel = panel.txt, main= 'Correlation Plot')

# Dimension Reduction
bike_data= subset(bike_data, select= -atemp)

###################### Model Development #######################

#Splitting the data into Training set and Test set
library(caTools)
split= sample.split(bike_data$cnt, SplitRatio= 0.8)
training_set= subset(bike_data, split== TRUE)
test_set= subset(bike_data, split== FALSE)

################### Multiple Linear Regression #############################
regressor= lm(formula = cnt ~., data = training_set)
# Adjusted R- square = 0.9842, MAPE= 4.5
#Predicting the test set results
y_pred= predict(regressor,test_set[, -14])


# Applying K-fold Cross Validation to deal with variance problem while testing the model with
# another test set
library(caret)
folds= createFolds(training_set$cnt, k= 10)
cv= lapply(folds, function(x) {
  training_fold= training_set[-x, ]
  test_fold= training_set[x, ]
  regressor= lm(formula = cnt ~., data = training_fold)
  y_pred= predict(regressor,test_fold[, -14])
  mape= function(y, yhat){
    mean(abs((y-yhat)/y))*100
  }
  error= mape(test_fold[,14], y_pred)
  return(error)
})
MAPE_LR=mean(as.numeric(cv))

# Creating function to calculate MAPE
mape= function(y, yhat){
  mean(abs((y-yhat)/y))*100
}

#alternate method
regr.eval(test_set[, 14], y_pred, stats = c('mae', 'rmse', 'mape', 'mse'))

# Evaluation: MAPE= 4.7, RMSE= 3.01

######################## Decision TREE ######################

regressor_DT= rpart(formula = cnt ~., data = training_set, method = "anova")
#Predicting the test set result
y_pred_DT= predict(regressor_DT, test_set[, -14])

mape(test_set[,14], y_pred_DT)

regr.eval(test_set[, 14], y_pred_DT, stats = c('mae', 'rmse', 'mape', 'mse'))

#Evaluation:- MAPE= 11.58, RMSE= 5.6
################## Random FOrest ###########################

regressor_RF= randomForest(cnt ~., training_set, importance= TRUE, ntree= 200)
#Predicting the test set result
y_pred_RF= predict(regressor_RF, test_set[,-14])

mape(test_set[,14], y_pred_RF)

regr.eval(test_set[, 14], y_pred_RF, stats = c('mae', 'rmse', 'mape', 'mse'))
# Evaluation:- Mape= 10.03, RMSE= 4.17

# We can clearly see the diffence among the three models that we developed.
#Multiple Linear Regression has the lowest vale of the Error Metrics and thus we will select multiple
#Linear regression Model as our final one. Also Applied K-fold cross validation to enhance the
#performance of the model.

# Extracting Predicted values from the Multiple Linear Regression Model

output = data.frame(test_set, pred_cnt = y_pred_RF)

write.csv(output, file = 'Predicted_bike_cnt.csv', row.names = FALSE, quote=FALSE)


