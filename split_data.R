library("stringr")
library("ggplot2")
options(stringsAsFactors = FALSE)

setwd("~/textanalytics/")

data_folder <- "data"

# Training Set data.
data<-read.csv(paste0(data_folder, "/", "Train_rev1.csv"),
               stringsAsFactors=FALSE)
data$DescriptionLength <- nchar(data$FullDescription)

# Examine the distribution of  Description Length
qplot(data$DescriptionLength, geom="histogram", xlab="Full 
      Text Description Length (number of characters)", main=
        "Histogram of Job Description Length")
# Examine quantiles. 
quantile(data$DescriptionLength, c(0, 0.01, 0.5, 0.99, 1.0))

# Look at the distribution of the log of Description Length.
qplot(log(data$DescriptionLength), geom="histogram")

# Tidy the FullDescription
# no_punctuation<-gsub("[/.,><():*0-9;&]", " ",
                     data$FullDescription)
# all_text<-paste(no_punctuation)
# words<- word(all_text)
# unique_words<-unique(words)

# Location rollup information. Also provided by Kaggle.
location_tree<- read.csv(paste0(data_folder, "/",
                                "Location_Tree.csv"),
                         stringsAsFactors=FALSE)

loc<-strsplit(location_tree[, 1], split="~")
dd<-do.call(rbind.data.frame, loc)
colnames(dd)<-paste0("Location", 1:7)

set.seed(23)
binom<-rbinom(nrow(data), 1, p=.2)

train_data <- data[binom==0,]
# Shuffle
train_data<-train_data[sample(nrow(train_data)),]

test_data <- data[binom==1,]
# Shuffe
test_data<-test_data[sample(nrow(test_data)),]


setwd("data/")
# For writing scripts with limited memory usage for fast iterations.
write.csv(train_data[1:50,], "train_mini.csv", row.names=FALSE)
write.csv(train_data[1:10000,], "train_medium.csv", row.names=FALSE)

write.csv(train_data, "train.csv", row.names=FALSE)
write.csv(test_data, "test.csv", row.names=FALSE)