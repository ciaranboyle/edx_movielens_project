##########################################################
# Install and load rpackages
##########################################################
if (!require("tidyverse")) {
  install.packages("tidyverse")
  library(tidyverse)
}
if (!require("caret")) {
  install.packages("caret")
  library(caret)
}
if (!require("data.table")) {
  install.packages("data.table")
  library(data.table)
}
if (!require("dplyr")) {
  install.packages("dplyr")
  library(dplyr)
}

##########################################################
# Generate EDX Dataset
##########################################################
#generate edx
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# Generate Training and Test Datasets
##########################################################
#generate training and test sets
set.seed(28, sample.kind="Rounding")
partition_index <- createDataPartition(y = edx$rating, times = 1, p = 0.7, list=FALSE)
training_edx <- edx[partition_index,]
temp <- edx[-partition_index,]
validation_edx <- validation

# Make sure userId and movieId in train set are also in test set
test_edx <- temp %>% 
  semi_join(training_edx, by = "movieId") %>%
  semi_join(training_edx, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_edx)
training_edx <- rbind(training_edx, removed)
rm(partition_index, temp, removed)
##########################################################
# Write RMSE function
##########################################################
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

##########################################################
# Compute the user average and calculate the RMSE
##########################################################
average_user_rating <- mean(training_edx$rating)
RMSE(average_user_rating,test_edx$rating)
##########################################################
# Compute the movie bias and calculate the RMSE
##########################################################
average_movie_rating <- training_edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - average_user_rating))

predicted_ratings <- test_edx %>%
  left_join(average_movie_rating, by="movieId") %>%
  mutate(pred = average_user_rating + b_m) %>%
  pull(pred)
RMSE(predicted_ratings, test_edx$rating)
##########################################################
# Compute the user bias and calculate the RMSE
##########################################################
user_bias <- training_edx %>% 
  left_join(average_movie_rating, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - average_user_rating - b_m))

##test RMSE
predicted_ratings <- test_edx %>%
  left_join(average_movie_rating, by="movieId") %>%
  left_join(user_bias, by="userId") %>%
  mutate(pred = average_user_rating + b_m + b_u) %>%
  pull(pred)
RMSE(predicted_ratings, test_edx$rating)


##########################################################
# Compute the movie-year bias and calculate the RMSE
##########################################################
year_bias <- training_edx %>%
  mutate(year_rated = year((as.POSIXct(training_edx$timestamp, origin="1970-01-01")))) %>% #used to convert timestamp into year and extract 
  left_join(average_movie_rating, by="movieId") %>%
  left_join(user_bias, by="userId") %>%
  group_by(movieId, year_rated) %>% #group entries by movieId as well as year in which it was rated
  summarize(b_y = mean(rating - average_user_rating - b_m - b_u))

##test RMSE
predicted_ratings_movieyear <- test_edx %>%
  mutate(year_rated = year((as.POSIXct(test_edx$timestamp, origin="1970-01-01")))) %>% 
  left_join(average_movie_rating, by="movieId") %>%
  left_join(user_bias, by="userId") %>%
  left_join(year_bias, by=c("movieId", "year_rated")) %>%
  replace(is.na(.), 0) %>%
  mutate(pred = average_user_rating + b_m + b_u + b_y) %>%
  pull(pred)
RMSE(predicted_ratings_movieyear, test_edx$rating)
##########################################################
# Compute the user-genre bias and calculate the RMSE
##########################################################
#create an index to filter out genres with less than 1000 ratings when calcualting the usergenre bias
usergenre_bias_index <- training_edx %>%
  mutate(year_rated = year((as.POSIXct(training_edx$timestamp, origin="1970-01-01")))) %>%  
  left_join(average_movie_rating, by="movieId") %>%
  left_join(user_bias, by="userId") %>%
  left_join(year_bias, by=c("movieId", "year_rated")) %>%
  replace(is.na(.), 0) %>%
  group_by(genres) %>%
  summarize(n=n(), index=1) %>% #counts observations per genre
  filter(n>=1000) %>% #filters out genres with less than 1000 ratings
  arrange(desc(n)) %>%
  select(-n)

usergenre_bias <- training_edx %>%
  mutate(year_rated = year((as.POSIXct(training_edx$timestamp, origin="1970-01-01")))) %>%  
  left_join(average_movie_rating, by="movieId") %>%
  left_join(user_bias, by="userId") %>%
  left_join(year_bias, by=c("movieId", "year_rated")) %>%
  replace(is.na(.), 0) %>%
  left_join(usergenre_bias_index, by="genres") %>%
  filter(index==1) %>% #filters out genres that are not in the index (i.e.: less than 1000 ratings)
  group_by(userId,genres) %>%
  summarize(b_ug = mean(rating - average_user_rating - b_m - b_u - b_y))

##test RMSE
predicted_ratings_usergenre <- test_edx %>%
  mutate(year_rated = year((as.POSIXct(test_edx$timestamp, origin="1970-01-01")))) %>% 
  left_join(average_movie_rating, by="movieId") %>%
  left_join(user_bias, by="userId") %>%
  left_join(year_bias, by=c("movieId", "year_rated")) %>%
  replace(is.na(.), 0) %>%
  left_join(usergenre_bias, by=c("userId","genres")) %>%
  replace(is.na(.), 0) %>%
  mutate(pred = average_user_rating + b_m + b_u + b_y + b_ug) %>%
  pull(pred)
RMSE(predicted_ratings_usergenre, test_edx$rating)
##########################################################


##########################################################
##########################################################
# Regularize the movie bias and calculate the RMSE
##########################################################
##generate test lambdas
lambdas <- seq(0,10,1)

##calculate rmses for different lambdas
moviebias_rmses <- sapply(lambdas, function(l){
  
  average_movie_rating_reg <- training_edx %>% 
    group_by(movieId) %>% 
    summarize(b_m = sum(rating - average_user_rating)/(n()+l))

  predicted_ratings <- test_edx %>%
    mutate(year_rated = year((as.POSIXct(test_edx$timestamp, origin="1970-01-01")))) %>% 
    left_join(average_movie_rating_reg, by="movieId") %>%
    left_join(user_bias, by="userId") %>%
    left_join(year_bias, by=c("movieId", "year_rated")) %>%
    replace(is.na(.), 0) %>%
    left_join(usergenre_bias, by=c("userId","genres")) %>%
    replace(is.na(.), 0) %>%
    mutate(pred = average_user_rating + b_m + b_u + b_y + b_ug) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_edx$rating))
})

#generate graph
movie_bias_chart <- ggplot(as.data.frame(cbind(lambdas,moviebias_rmses)), aes(lambdas, moviebias_rmses)) + geom_point() + 
  labs(title = "RMSE Scores per Lambda", subtitle = "") +
  xlab("Lambdas") + scale_x_discrete(limits=c(0:10)) +
  ylab("RMSE Score")

#check optimal lambda
lambda_moviebias <- lambdas[which.min(moviebias_rmses)]
lambda_moviebias

##incorporate optimal lambda rating into movie bias
average_movie_rating_reg <- training_edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = sum(rating - average_user_rating)/(n()+2))

#check RMSE score for optimal lambda
moviebias_rmses[which.min(moviebias_rmses)]


##########################################################
# Regularize the user bias and calculate the RMSE
##########################################################
##generate test lambdas
lambdas <- seq(0,10,1)

##calculate rmses for different lambdas
userbias_rmses <- sapply(lambdas, function(l){
  
  user_bias_reg <- training_edx %>% 
    left_join(average_movie_rating, by = "movieId") %>%
    group_by(userId) %>% 
    summarize(b_u = sum(rating - average_user_rating - b_m)/(n()+l))
  
  predicted_ratings <- test_edx %>%
    mutate(year_rated = year((as.POSIXct(test_edx$timestamp, origin="1970-01-01")))) %>% 
    left_join(average_movie_rating_reg, by="movieId") %>%
    left_join(user_bias_reg, by="userId") %>%
    left_join(year_bias, by=c("movieId", "year_rated")) %>%
    replace(is.na(.), 0) %>%
    left_join(usergenre_bias, by=c("userId","genres")) %>%
    replace(is.na(.), 0) %>%
    mutate(pred = average_user_rating + b_m + b_u + b_y + b_ug) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_edx$rating))
})

#generate graph
user_bias_chart <- ggplot(as.data.frame(cbind(lambdas,userbias_rmses)), aes(lambdas, userbias_rmses)) + geom_point() + 
  labs(title = "RMSE Scores per Lambda", subtitle = "") +
  xlab("Lambdas") + scale_x_discrete(limits=c(0:10)) +
  ylab("RMSE Score")
user_bias_chart

#check optimal lambda
lambda_userbias <- lambdas[which.min(userbias_rmses)]
lambda_userbias

##incorporate optimal lambda rating into user bias
user_bias_reg <- training_edx %>% 
  left_join(average_movie_rating, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - average_user_rating - b_m)/(n()+5))

#check RMSE score for optimal lambda
userbias_rmses[which.min(userbias_rmses)]

##########################################################
# Regularize the movie-year bias and calculate the RMSE
##########################################################
##generate test lambdas
lambdas <- seq(0,100,10)

##calculate rmses for different lambdas
movieyear_rmses <- sapply(lambdas, function(l){
  
  year_bias_reg <- training_edx %>%
    mutate(year_rated = year((as.POSIXct(training_edx$timestamp, origin="1970-01-01")))) %>%  
    left_join(average_movie_rating, by="movieId") %>%
    left_join(user_bias_reg, by="userId") %>%
    group_by(movieId, year_rated) %>% 
    summarize(b_y = sum(rating - average_user_rating - b_m - b_u)/(n()+l))
  
  predicted_ratings <- test_edx %>%
    mutate(year_rated = year((as.POSIXct(test_edx$timestamp, origin="1970-01-01")))) %>% 
    left_join(average_movie_rating_reg, by="movieId") %>%
    left_join(user_bias_reg, by="userId") %>%
    left_join(year_bias_reg, by=c("movieId", "year_rated")) %>%
    replace(is.na(.), 0) %>%
    left_join(usergenre_bias, by=c("userId","genres")) %>%
    replace(is.na(.), 0) %>%
    mutate(pred = average_user_rating + b_m + b_u + b_y + b_ug) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_edx$rating))
})

#generate graph
movieyear_chart <- ggplot(as.data.frame(cbind(lambdas,movieyear_rmses)), aes(lambdas, movieyear_rmses)) + geom_point() + 
  labs(title = "RMSE Scores per Lambda", subtitle = "") +
  xlab("Lambdas") + scale_x_discrete(limits=seq(0,100,10)) +
  ylab("RMSE Score")
movieyear_chart

#check optimal lambda
lambda_movieyear <- lambdas[which.min(movieyear_rmses)]
lambda_movieyear

##incorporate optimal lambda rating into movie year bias
year_bias_reg <- training_edx %>%
  mutate(year_rated = year((as.POSIXct(training_edx$timestamp, origin="1970-01-01")))) %>%  
  left_join(average_movie_rating, by="movieId") %>%
  left_join(user_bias_reg, by="userId") %>%
  group_by(movieId, year_rated) %>% 
  summarize(b_y = sum(rating - average_user_rating - b_m - b_u)/(n()+40))

#check RMSE score for optimal lambda
movieyear_rmses[which.min(movieyear_rmses)]

##########################################################
# Regularize the user-genre bias and calculate the RMSE
##########################################################
##generate test lambdas
lambdas <- seq(0,10,1)

##calculate rmses for different lambdas
usergenre_rmses <- sapply(lambdas, function(l){
  
  usergenre_bias_index <- training_edx %>%
    mutate(year_rated = year((as.POSIXct(training_edx$timestamp, origin="1970-01-01")))) %>%  
    left_join(average_movie_rating_reg, by="movieId") %>%
    left_join(user_bias_reg, by="userId") %>%
    left_join(year_bias_reg, by=c("movieId", "year_rated")) %>%
    replace(is.na(.), 0) %>%
    group_by(genres) %>%
    summarize(n=n(), index=1) %>%
    filter(n>=1000) %>%
    arrange(desc(n)) %>%
    select(-n)
  
  usergenre_bias_reg <- training_edx %>%
    mutate(year_rated = year((as.POSIXct(training_edx$timestamp, origin="1970-01-01")))) %>%  
    left_join(average_movie_rating_reg, by="movieId") %>%
    left_join(user_bias_reg, by="userId") %>%
    left_join(year_bias_reg, by=c("movieId", "year_rated")) %>%
    replace(is.na(.), 0) %>%
    left_join(usergenre_bias_index, by="genres") %>%
    filter(index==1) %>%
    group_by(userId,genres) %>%
    summarize(b_ug = sum(rating - average_user_rating - b_m - b_u - b_y)/(n()+l))
  
  predicted_ratings <- test_edx %>%
    mutate(year_rated = year((as.POSIXct(test_edx$timestamp, origin="1970-01-01")))) %>% 
    left_join(average_movie_rating_reg, by="movieId") %>%
    left_join(user_bias_reg, by="userId") %>%
    left_join(year_bias_reg, by=c("movieId", "year_rated")) %>%
    replace(is.na(.), 0) %>%
    left_join(usergenre_bias_reg, by=c("userId","genres")) %>%
    replace(is.na(.), 0) %>%
    mutate(pred = average_user_rating + b_m + b_u + b_y + b_ug) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_edx$rating))
})

#generate graph
usergenre_chart <- ggplot(as.data.frame(cbind(lambdas,usergenre_rmses)), aes(lambdas, usergenre_rmses)) + geom_point() + 
  labs(title = "RMSE Scores per Lambda", subtitle = "") +
  xlab("Lambdas") + scale_x_discrete(limits=seq(0,10,1)) +
  ylab("RMSE Score")
usergenre_chart

#check optimal lambda
lambda_usergenre <- lambdas[which.min(usergenre_rmses)]
lambda_usergenre

##incorporate optimal lambda rating into user genre bias
usergenre_bias_reg <- training_edx %>%
  mutate(year_rated = year((as.POSIXct(training_edx$timestamp, origin="1970-01-01")))) %>%  
  left_join(average_movie_rating_reg, by="movieId") %>%
  left_join(user_bias_reg, by="userId") %>%
  left_join(year_bias_reg, by=c("movieId", "year_rated")) %>%
  replace(is.na(.), 0) %>%
  left_join(usergenre_bias_index, by="genres") %>%
  filter(index==1) %>%
  group_by(userId,genres) %>%
  summarize(b_ug = sum(rating - average_user_rating - b_m - b_u - b_y)/(n()+8))

##check RMSE score
usergenre_rmses[which.min(usergenre_rmses)]
##########################################################


##########################################################
# Calculate RMSE for validation set
##########################################################
predicted_ratings_validation <- validation_edx %>%
  mutate(year_rated = year((as.POSIXct(validation_edx$timestamp, origin="1970-01-01")))) %>% 
  left_join(average_movie_rating_reg, by="movieId") %>%
  left_join(user_bias_reg, by="userId") %>%
  left_join(year_bias_reg, by=c("movieId", "year_rated")) %>%
  replace(is.na(.), 0) %>%
  left_join(usergenre_bias_reg, by=c("userId","genres")) %>%
  replace(is.na(.), 0) %>%
  mutate(pred = average_user_rating + b_m + b_u + b_y + b_ug) %>%
  pull(pred)
RMSE(predicted_ratings_validation, validation_edx$rating)




