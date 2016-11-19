#source("http://bioconductor.org/biocLite.R")
#biocLite("rhdf5")

library(rhdf5)
library(NLP)
library(lda)
library(LDAvis)
library(tm)
library(topicmodels)
library(ClustOfVar)
library(MASS)
library(pscl)
library(lme4)
library(gbm)

setwd("~/GitHub/Fall2016-proj4-daisy0311")

###read in lyric and id dataset
load("./data/lyr.RData")
id <- read.table("./data/common_id.txt")
n <- nrow(id)
rownames(lyr) <- lyr[,1]
lyr <- lyr[,-1]

###splite into train and test dataset
testindex <- sample(n, trunc(n/5))

lyr.train <- lyr[-testindex,]
lyr.test <- lyr[testindex,]

###############################################
##########feature extraction###################
###############################################

h5ls("./data/data/A/T/C/TRATCDG128F932B4D2.h5")

path = "./data/data/"
loud.mean <- rep(NA, n)
pitch.mean <- timbre.mean <- matrix(NA, n, 12)
time.sig <- nsection <- loud.med <- loud.sd <- rep(NA, n)
loud.start.mean <- loud.start.med <- loud.start.sd <- rep(NA, n)
pitch.med <- pitch.sd <- timbre.med <- timbre.sd <- matrix(NA, n, 12)

file.names <- dir(path, pattern =".h5", recursive = T)
for(i in 1:length(file.names)){
  file <- h5read(paste(path, file.names[i], sep = ''), "/analysis")
  time.sig[i] <- length(file$beats_start)/length(file$bars_start)
  nsection[i] <- length(file$sections_start)
  loud.mean[i] <- mean(file$segments_loudness_max)
  loud.med[i] <- median(file$segments_loudness_max)
  loud.sd[i] <- sd(file$segments_loudness_max)
  loud.start.mean[i] <- mean(file$segments_loudness_start)
  loud.start.med[i] <- median(file$segments_loudness_start)
  loud.start.sd[i] <- sd(file$segments_loudness_start)
  pitch.mean[i,] <- apply(file$segments_pitches, 1, mean)
  pitch.med[i,] <- apply(file$segments_pitches, 1, median)
  pitch.sd[i,] <- apply(file$segments_pitches, 1, sd)
  timbre.mean[i,] <- apply(file$segments_timbre, 1, mean)
  timbre.med[i,] <- apply(file$segments_timbre, 1, median)
  timbre.sd[i,] <- apply(file$segments_timbre, 1, sd)
}

time.sig[is.na(time.sig)] <- mean(time.sig, na.rm = T)
time.sig[time.sig==Inf] <- mean(time.sig[time.sig!=Inf])

X <- cbind(time.sig, nsection, loud.mean, loud.med, loud.sd, 
                loud.start.mean, loud.start.med, loud.start.sd, 
                pitch.mean, pitch.med, pitch.sd, 
                timbre.mean,timbre.med, timbre.sd)
#X <- cbind(loudness, pitch, timbre)

X.train <- X[-testindex,]
X.test <- X[testindex,]

save(X.train, X.test, X, testindex, lyr.train, lyr.test, file = "splite3.RData")

###################################
########## baseline ###############
###################################

f.word <- colSums(lyr.train)
rank.base <- rank(-f.word)
r.bar <- mean(rank.base)

rs.base <- apply(lyr.test, 1, function(x){
  index <- which(x!=0)
  rs <- sum(rank.base[index])/r.bar.base/length(index)
  return(rs)
})

mean(rs.base)

###############################################
##########feature clustering###################
###############################################


clu <- hclustvar(t(cbind(loudness, pitch, timbre)))
plot(clu)
stab <- stability(clu, B = 100)
#plot(stab,main="Stability of the partitions")
part <- cutreevar(clu, 5)
summary(part)
part$scores


km <- kmeans(X, 20)
km$cluster

###############################################
############## glm regression #################
###############################################

pred <- matrix(NA, length(testindex), 5000)

for (i in 1:5000){
  X.i <- data.frame(X.train, wordi = lyr.train[,i])
  fit <- glm(wordi ~ ., data = X.i, family = "poisson")
  pred[,i] <- predict(fit, newdata = data.frame(X.test))
}

###calculate predictive rank sum
rs.glm <- rep(NA, length(testindex))

for (i in 1:length(testindex)){
  rank.glm <- rank(-pred[i,])
  index <- which(lyr.test[i,] != 0)
  rs.glm[i] <- sum(rank.glm[index])/r.bar/length(index)
}

mean(rs.glm)
save(pred, rs.glm, file = "./glmoutput.RData")

###############################################
############## gbm regression #################
###############################################

source("./lib/train_gbm.R")

pred.gbm <- matrix(NA, length(testindex), 5000)
notzero.index <- which(f.word!=0)

for (i in notzero.index){
  fit.gbm <- train_gbm(X.train, lyr.train[,i])
  pred.gbm[,i] <- predict(fit.gbm$fit, newdata=data.frame(X.test), 
                      n.trees=fit.gbm$iter, type="response")
}

save(pred.gbm, file="gbm.500.to5000.1118.RData")

###calculate predictive rank sum
rs.gbm <- rep(NA, length(testindex))

for (i in 1:length(testindex)){
  rank.gbm <- rank(-pred.gbm[i,])
  index <- which(lyr.test[i,] != 0)
  rs.gbm[i] <- sum(rank.gbm[index])/r.bar/length(index)
}

mean(rs.gbm)

#####################################
############## svm  #################
#####################################

source("./lib/train_svm.R")

pred.svm <- matrix(NA, length(testindex), 5000)
notzero.index <- which(f.word!=0)
par <- list(kernel = "radial", cost = 1, gamma = 0.001)

for (i in notzero.index){
  fit.svm <- train_svm(X.train, lyr.train[,i], par)
  pred.svm[,i] <- predict(fit.svm, newdata=data.frame(X.test))
}

###calculate predictive rank sum
rs.svm <- rep(NA, length(testindex))

for (i in 1:length(testindex)){
  rank.svm <- rank(-pred.svm[i,])
  index <- which(lyr.test[i,] != 0)
  rs.svm[i] <- sum(rank.svm[index])/r.bar/length(index)
}

mean(rs.svm)


###############################################
##########lyrics topic modeling################
###############################################

########## lda.collapsed.gibbs.sampler

stop.words <- stopwords("English")

del <- colnames(lyr) %in% stop_words | f_words < 5
term.table <- f_words[!del]
vocab_m <- names(term.table)
index <- match(vocab_m, colnames(lyr))

lyr_cleaned<-lyr[,index]
trackid <- lyr[,1]
rownames(lyr_cleaned) <- trackid


#pre
# read in some stopwords:

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <-which(x != 0)
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}

documents <- apply(lyr_cleaned,1, get.terms)

##
D <- length(documents)  # number of documents (2,000)
W <- length(vocab_m)  # number of terms in the vocab (14,568)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (546,827)
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus [8939, 5544, 2411, 2410, 2143, ...]

##
K <- 10
G <- 5000
alpha <- 0.02
eta <- 0.02


fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab_m, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)

##
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

##
Lyrics <- list(phi = phi,
               theta = theta,
               doc.length = doc.length,
               vocab = vocab_m,
               term.frequency = term.frequency)

mylyr <- mylyr[,-which(colSums(lyr)==0)]


############# CTM method

r1 <- apply(lyr, 1, function(x) return(which.max(x)))
sort(table(colnames(lyr)[r1]))
plot(colSums(lyr))
colnames(lyr)[which(colSums(lyr) > 2350)]

ctm <- CTM(lyr, 20)
dim(posterior(ctm)$topics)
terms(ctm)
topics(ctm)[6]
posterior(ctm)$topics[6,]

ctm10 <- CTM(lyr, 10)
ctm15 <- CTM(lyr, 15)



#################################
########### testing #############
#################################

path = "./data/TestSongFile100/"
n <- 100
loud.mean <- rep(NA, n)
pitch.mean <- timbre.mean <- matrix(NA, n, 12)
time.sig <- nsection <- loud.med <- loud.sd <- rep(NA, n)
loud.start.mean <- loud.start.med <- loud.start.sd <- rep(NA, n)
pitch.med <- pitch.sd <- timbre.med <- timbre.sd <- matrix(NA, n, 12)

file.names <- dir(path, pattern =".h5", recursive = T)
for(i in 1:length(file.names)){
  file <- h5read(paste(path, file.names[i], sep = ''), "/analysis")
  time.sig[i] <- length(file$beats_start)/length(file$bars_start)
  nsection[i] <- length(file$sections_start)
  loud.mean[i] <- mean(file$segments_loudness_max)
  loud.med[i] <- median(file$segments_loudness_max)
  loud.sd[i] <- sd(file$segments_loudness_max)
  loud.start.mean[i] <- mean(file$segments_loudness_start)
  loud.start.med[i] <- median(file$segments_loudness_start)
  loud.start.sd[i] <- sd(file$segments_loudness_start)
  pitch.mean[i,] <- apply(file$segments_pitches, 1, mean)
  pitch.med[i,] <- apply(file$segments_pitches, 1, median)
  pitch.sd[i,] <- apply(file$segments_pitches, 1, sd)
  timbre.mean[i,] <- apply(file$segments_timbre, 1, mean)
  timbre.med[i,] <- apply(file$segments_timbre, 1, median)
  timbre.sd[i,] <- apply(file$segments_timbre, 1, sd)
}

time.sig[is.na(time.sig)] <- mean(time.sig, na.rm = T)
time.sig[time.sig==Inf] <- mean(time.sig[time.sig!=Inf])

X.final <- cbind(time.sig, nsection, loud.mean, loud.med, loud.sd, 
           loud.start.mean, loud.start.med, loud.start.sd, 
           pitch.mean, pitch.med, pitch.sd, 
           timbre.mean,timbre.med, timbre.sd)


pred.final <- matrix(NA, 100, 5000)
notzero.index <- which(colSums(lyr)!=0)

for (i in notzero.index){
  fit.gbm <- train_gbm(X, lyr[,i])
  pred.final[,i] <- predict(fit.gbm$fit, newdata=data.frame(X.final), 
                          n.trees=fit.gbm$iter, type="response")
}

save(pred.final, file="gbm.final.test.RData")
write.csv(pred.final, file = "pred.result.csv")

pred.rank <- matrix(NA, 100, 5000)
wordi <- c(3:4, 30:5000)

for (i in 1:100){
  pred.rank[i, wordi] <- rank(-pred.final[i, wordi])
}

pred.rank[,-wordi] <- 4987

write.csv(pred.rank, file = "pred.rank.fix.csv")




