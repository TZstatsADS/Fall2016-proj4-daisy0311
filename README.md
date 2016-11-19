# Project: Words 4 Music

### [Project Description](doc/Project4_desc.md)

![image](http://cdn.newsapi.com.au/image/v1/f7131c018870330120dbe4b73bb7695c?width=650)

Term: Fall 2016

+ [Data link](https://courseworks2.columbia.edu/courses/11849/files/folder/Project_Files?preview=763391)-(**courseworks login required**)
+ [Data description](doc/readme.html)
+ Contributor's name: Minghao Dai
+ Projec title: Words 4 Music
+ Project summary: 
  
  In this project, I worked individually on the prediction of the lyric rank from the music features. 
  
  1. First splite the data into train set (1880 songs) and test set (470 songs). 
  
  2. Calculate a baseline predictive rank sum by using the train marginal word ranks as the prediction for any songs in the test set. The baseline is about 0.2558. 
  
  3. The first task is doing the feature selection from the 2350 .h5 files. From the feature description list (under doc folder), I noticed that the most relevant features should be "loudness", "pitch" and "timbre". So I run a pilot regression on the column mean of these three features (which has 25 variables in total) and found that the result is almost the same as the baseline. Then I included the time signature (# beats/# bars), nsection, and mean, median, sd of all the "loudness", "pitch" and "timbre" related variables (use column mean, median, sd if it's a matrix) as my final features. 
  
  4. Then I tried kmeans and hierarchical cluster methods to do dimension reduction for the feature matrix X. Since the most of the variables can not be seperate completly, and the variable number 80 is not too big relative to the sample size 2350. I chose to use the original feature matrix for my model. 
  
  5. From the lyric perspective, I tried to use topic model to split the songs into several topics in order to reduce regression times. I used lda.collapsed.gibbs.sampler() and CTM() functions in topicmodels package respectively. 
  
  6. To find the relationship between lyrics and music features. I tried glm model for binomial (logistic regression), poisson and negative binomial for each of the words seperately. 
  
  7. I also used gradient boosting regression which performed well in the last project. I specify the distribution as poisson to be consistent with the count numbers in the lyric matrix. To tuning the parameters and control the training time, I just run it on random chosen testing set and don't do cross validation which is very time consuming. 
  
  8. Finally I tried support vector machine to predict each lyric words count, the result is not better than gbm. So I decided to use gbm for the final model. 


Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
