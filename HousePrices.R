houseData <- read.csv("train.csv")
houseData[is.na(houseData)] = 0
plot(houseData)
summary(houseData)
dim(houseData)
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")
tensorflow::install_tensorflow()
tensorflow::tf_config()

install.packages("magrittr")
install.packages("tidyverse")
install.packages("keras")
install.packages("caret")
library(magrittr)
library(tidyverse)
library(keras)
library(caret)

View(train_clean)
View(test_clean)
train <- read.csv("train.csv")
head(train)

y <- train %>% select(SalePrice)
train <- train %>% select(-SalePrice)

test <- read.csv("test.csv")
head(test)

train %>% 
  select(everything()) %>%  # replace to your needs
  summarise_all(funs(sum(is.na(.))))

test %>% 
  select(everything()) %>%  # replace to your needs
  summarise_all(funs(sum(is.na(.))))

common_features <- intersect(colnames(train %>% select_if(~ !any(is.na(.)))),
                             colnames(test %>% select_if(~ !any(is.na(.)))))  
common_features

train_clean <- train[,common_features] %>% select(-Id)
test_clean <- test[,common_features] %>% select(-Id)

colnames(train_clean)
colnames(test_clean)


print("Numerical variables")
colnames(train_clean %>% select_if(is.numeric))

print("Categorical variables")
setdiff(colnames(train_clean), colnames(train_clean %>% select_if(is.numeric)))

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

train_clean <- train_clean %>% #
  mutate_if(is.character, encode_ordinal)

test_clean <- test_clean %>% #
  mutate_if(is.character, encode_ordinal)

process <- preProcess(train_clean, method=c("range"))

train_scale <- predict(process, train_clean)
head(train_scale)

test_scale <- predict(process, test_clean)
head(test_scale)

model = keras_model_sequential() %>% 
  layer_dense(units=64, 
              activation="relu", 
              kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01),
              input_shape=c(ncol(train_scale))) %>% 
  layer_dense(units=64, 
              activation = "relu",
              kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>%
  layer_dense(units=1, activation="linear")


model %>% compile(
  loss = "mse",
  optimizer =  "rmsprop", 
  metrics = list(c("mean_absolute_error")) # ,"root_mean_squared_error"
)

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)

history <- model %>% fit(
  x = as.matrix(train_scale),
  y = as.matrix(y$SalePrice),
  epochs = 500,
  validation_split = 0.25,
  batch_size = 64,
  verbose = 1,
  callbacks = list(print_dot_callback)
)

plot(history)

test_preds <- model %>% predict(as.matrix(test_scale))
head(test_preds)

submission <- read_csv("sample_submission.csv")

submission$SalePrice <- test_preds
head(submission)
View(submission)
write.csv(submission, 'submission.csv', row.names = FALSE)
