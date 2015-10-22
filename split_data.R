setwd("~/textanalytics/")
data<-read.csv("Train_rev1.csv",stringsAsFactors=FALSE)

set.seed(23)
binom<-rbinom(nrow(data), 1, p=.2)

train_data <- data[binom==0,]
test_data <- data[binom==1,]


write.csv(train_data[1:10,], "train_mini.csv", row.names=FALSE)

write.csv(train_data, "train.csv", row.names=FALSE)
write.csv(test_data, "test.csv", row.names=FALSE)