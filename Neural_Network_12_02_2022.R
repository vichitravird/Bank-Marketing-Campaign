####################Installing Libraries#######
install.packages("neuralnet")
library(neuralnet)
install.packages("caret")
library(caret)

####################Importing Data#################
bank.df <- read.csv("bank_bin.csv")

###############################Partitioning Data##############
set.seed(2)
train.index <- sample(c(1:dim(bank.df)[1]), dim(bank.df)[1]*0.6)  
train.df <- bank.df[train.index, ]
valid.df <- bank.df[-train.index, ]

##################Applying Neural Networks############
nn <- neuralnet(y ~ duration+month+poutcome+day+housing+age+pdays+campaign, data = train.df, linear.output = F, hidden = 3)

plot(nn, rep="best")

nn.pred <- predict(nn, valid.df, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred.classes), as.factor(valid.df$y))

nn <- neuralnet(y ~ education+balance+marital, data = train.df, linear.output = F, hidden = 3)

plot(nn, rep="best")

nn.pred <- predict(nn, valid.df, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred.classes), as.factor(valid.df$y))

nn <- neuralnet(y ~ duration+education+balance+marital, data = train.df, linear.output = F, hidden = 3)

plot(nn, rep="best")

nn.pred <- predict(nn, valid.df, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred.classes), as.factor(valid.df$y))


nn <- neuralnet(y ~ duration+education+balance+marital+age, data = train.df, linear.output = F, hidden = 3)

plot(nn, rep="best")

nn.pred <- predict(nn, valid.df, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred.classes), as.factor(valid.df$y))















