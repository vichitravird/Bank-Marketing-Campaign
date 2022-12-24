##############################Calling Libraries#############################################################################################################
install.packages("factoextra")
install.packages("tidyverse")
install.packages("purrr")

library(gmodels) # Cross Tables [CrossTable()]
library(ggmosaic) # Mosaic plot with ggplot [geom_mosaic()]
library(corrplot) # Correlation plot [corrplot()]
library(ggpubr) # Arranging ggplots together [ggarrange()]
library(cowplot) # Arranging ggplots together [plot_grid()]
library(caret) # ML [train(), confusionMatrix(), createDataPartition(), varImp(), trainControl()]
library(ROCR) # Model performance [performance(), prediction()]
library(plotROC) # ROC Curve with ggplot [geom_roc()]
library(pROC) # AUC computation [auc()]
library(PRROC) # AUPR computation [pr.curve()]
library(rpart) # Decision trees [rpart(), plotcp(), prune()]
library(rpart.plot) # Decision trees plotting [rpart.plot()]
library(ranger) # Optimized Random Forest [ranger()]
library(lightgbm) # Light GBM [lgb.train()]
library(xgboost) # XGBoost [xgb.DMatrix(), xgb.train()]
library(MLmetrics) # Custom metrics (F1 score for example)
library(tidyverse) # Data manipulation
library(doMC) # Parallel processing
library(ggpubr)
library(factoextra)
library(purrr)

require(multcomp)
require(gmodels)
require(Sleuth3) 
require(mosaic)
require(knitr)
#############################Importing Data#######################################################################################################
bank_data = read.csv(file = "bank_bin.csv", stringsAsFactors = F)

dim(bank_data)

#The dataset has 4521 rows and 17 columns.

names(bank_data)
##############################Exploraotry Data Analysis#############################################################################################################
favstats <- favstats(~age, data=bank_data); favstats
#min Q1 median Q3 max    mean       sd    n missing
# 19 33     39 49  87 41.1701 10.57621 4521       0

hist(bank_data$age, col='darkslategray1') #Left Skew
#Age is not normally distributed

#Performing similar set of analysis on the attribute: balance
favstats <- favstats(~balance, data=bank_data); favstats
hist(bank_data$balance, col='darkslategray1') #Right Skew
#Age is not normally distributed

#Building Scatter Plots
## scatter plot with axes names
plot(bank_data$age~bank_data$balance)

#This is an unbalanced two-levels categorical variable, 
#88.5% of values taken are “no” (or “0”) and only 11.5% of the values are “yes” (or “1”). 
#It is more natural to work with a 0/1 dependent variable:

bank_data=bank_data %>%
  mutate(y=factor(if_else(y=="yes", "1", "0"),
                  levels=c("0", "1")))
head(bank_data)

sum(is.na(bank_data))

#There’s no missing value in the dataset. 
#However, according to the data documentation, “unknown” value means NA.

sum(bank_data == "unknown")

#There are 12,718 unknown values in the dataset, let’s try to
#find out which variables suffer the most from those “missing values”.


bank_data %>% 
  summarise_all(list(~sum(. == "unknown"))) %>% 
  gather(key = "variable", value = "num_unknown") %>% 
  arrange(-num_unknown)

#4 features have at least 1 unknown value. 
#Before deciding how to manage those missing values, we’ll study each variable
#and take a decision after visualisations. 
#We can’t afford to delete these many rows in our dataset, it’s more than 20% of our observations.

##############################Kmeans Clustering#################################################

# compute Euclidean distance
# (to compute other distance measures, change the value in method = manhattan)
d <- dist(bank_data, method = "euclidean")

# normalize input variables
bank_data_norm <- sapply(bank_data, scale)

# compute normalized distance based on variables Sales (large scale) and Fuel Cost (small scale)
d_norm <- dist(bank_data_norm, method = "euclidean")

# run kmeans algorithm 
set.seed(2)
km <- kmeans(d_norm, 4)

# show cluster membership
km$cluster



#### Table 15.10
# centroids
km$centers
km$withinss
km$size


#### Figure 15.5

# plot an empty scatter plot
plot(c(0), xaxt = 'n', ylab = "", type = "l", 
     ylim = c(min(km$centers), max(km$centers)), xlim = c(0, 16))

# label x-axes
axis(1, at = c(1:16), labels = names(bank_data_norm))

# plot centroids
for (i in c(1:4))
  lines(km$centers[i,], lty = i, lwd = 2, col = c("black", "dark grey","grey", "blue")
                                                       )

# name clusters
text(x =0.25, y = km$centers[, 2], labels = paste("Cluster", c(1:4)))

##############################Kmeans advanced###########################################

set.seed(1)
bank_data_norm <- sapply(bank_data, scale)
res.km <- kmeans(bank_data_norm, 2)
# K-means clusters showing the group of each individuals
res.km$cluster
res.km$size

fviz_cluster(res.km, data =bank_data_norm,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

set.seed(1)
bank_data_norm <- sapply(bank_data, scale)
res.km <- kmeans(bank_data_norm, 3)
# K-means clusters showing the group of each individuals
res.km$cluster

fviz_cluster(res.km, data =bank_data_norm,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)



bank_data_norm <- sapply(bank_data, scale)
res.km <- kmeans(bank_data_norm, 4)
# K-means clusters showing the group of each individuals
res.km$cluster

fviz_cluster(res.km, data =bank_data_norm,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

bank_data_norm <- sapply(bank_data, scale)
res.km <- kmeans(bank_data_norm, 5)
# K-means clusters showing the group of each individuals
res.km$cluster

fviz_cluster(res.km, data =bank_data_norm,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800","black", "pink"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)
############## Elbow Method ##############
bank_data_norm <- sapply(bank_data, scale)
wss<- function(k){kmeans(bank_data_norm, k, nstart=25)$tot.withinss}

#compute and plot wss for k =1 to k=15
k.values<-1:15

#Extract wss for 2-15 clusters
wss_values<-map_dbl(k.values, wss)
plot(k.values, wss_values,
     type="b", pch=19, frame=FALSE,
     xlab="Number of Clusters K",
     ylab="Total within-clusters sum of squares")

fviz_nbclust(bank_data, kmeans, method="wss")+geom_vline(xintercept=4, linetype=2)

############## Making Conclusion ##########
Matrix_x <- matrix(unlist(res.km))
write.csv(Matrix_x, "test_cluster_count_4.csv")
