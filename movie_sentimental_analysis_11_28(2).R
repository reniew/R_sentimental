#data = read.delim('./data/labeledTrainData.tsv', header = 1, quote = "",as.is=T)
#attach(data) # id, review, sentiment

install.packages("text2vec")
install.packages("tm")
install.packages("softmaxreg")
install.packages("h2o")
install.packages("dplyr")
install.packages("fastTextR")
install.packages("randomForest")
library("text2vec")
library("tm")
library("softmaxreg")
library("dplyr")
library("fastTextR")
library("randomForest")


data("movie_review")
review = movie_review$review
label = movie_review$sentiment
head(review)

review = gsub("<br />", "", review)
review = gsub("[^a-zA-Z /s]", "", review) # remove all character without alphabet&space
review = gsub("[^a-zA-Z]", " ", review) # remove special character beside space
review = tolower(review)
head(review)

tokenized_review = word_tokenizer(review)
head(tokenized_review)


stopword = stopwords()

'%nin%' <- Negate('%in%')
tokenized_review = lapply(tokenized_review, function(x){
  x[x %nin% stopword]
  
})


clean_review = c()

for (i in 1:length(tokenized_review)) {
  temp = tokenized_review[[i]]
  clean_review = append(clean_review, paste(temp, collapse = " "))
}

# document to vector

data("word2vec")
x = list()

result = matrix(ncol = 20)

for (one_review in clean_review) {
  embed_temp = wordEmbed(object = one_review, dictionary = word2vec, meanVec = TRUE)
  result = rbind(result, embed_temp)
}


train_data = data.frame(x = result[2:4801,], y = label[1:4800])

x_test = data.frame(x = result[4802:5001,])
y_test = data.frame(y = label[4801:5000])

len_test = length(y_test$y)

linear_model = lm(y ~ ., data = train_data)
output = predict(linear_model, x_test)
prediction = data.frame(y = round(output))
print(paste("Accuracy of linear model: ",(sum(prediction == y_test)/len_test)*100, "%"))

glm_model = glm.fit(y ~ ., data = train_data)
output = predict(glm_model, x_test)
prediction = data.frame(y = round(output))
print(paste("Accuracy of linear model: ",(sum(prediction == y_test)/len_test)*100, "%"))

rf_model = randomForest(y ~ ., data=train_data, ntree = 300)
output_rf = predict(rf_model, x_test)
prediction_rf = data.frame(y = round(output_rf))
print(paste("Accuracy of random forest model: ",(sum(prediction_rf == y_test)/len_test)*100, "%"))
as.numeric(output_rf>mean(output_rf))
print(paste("Accuracy of random forest model: ",(sum(as.numeric(output_rf>mean(output_rf)) == y_test)/len_test)*100, "%"))

library(h2o)

h2o.init()
dl_data <- as.h2o(train_data)
dl_model <- h2o.deeplearning(x = 1:20, y = 21,
                             training_frame = dl_data,
                             hidden = c(1000,100,1),
                             epochs = 100)

prediction_dl = h2o.predict(dl_model, as.h2o(x_test))
prediction_dl = round(as.data.frame(prediction_dl))
print(paste("Accuracy of deep learning model: ",(sum(prediction_dl == y_test)/len_test)*100, "%"))


#################################################################################
############################ train embedding vector #############################
#################################################################################

cntrl <- ft.control(word_vec_size = 50L, learning_rate = 0.1, max_len_ngram = 2L, 
                    min_count = 1L, nbuckets = 10000000L, epoch = 20L, nthreads = 20L)

model <- fasttext(input = "clean_review.txt", method = "skipgram", control = cntrl)

help("fasttext")

fast2vec = data.frame(get_word_vectors(model, words))
words_frame = data.frame(word = words)
embedding_data = bind_cols(words_frame, fast2vec)


result2 = matrix(ncol = 50)

for (one_review in clean_review) {
  embed_temp = wordEmbed(object = one_review, dictionary = embedding_data, meanVec = TRUE)
  result2 = rbind(result2, embed_temp)
}

train_data2 = data.frame(x = result2[2:4801,], y = label[1:4800])

x_test2 = data.frame(x = result2[4802:5001,])

linear_model = lm(y ~ ., data = train_data2)
output = predict(linear_model, x_test2)
prediction = data.frame(y = round(output))
print(paste("Accuracy of linear model: ",(sum(prediction == y_test)/len_test)*100, "%"))

glm_model = glm(y ~ ., data = train_data2)
output = predict(glm_model, x_test2)
prediction = data.frame(y = round(output))
print(paste("Accuracy of linear model: ",(sum(prediction == y_test)/len_test)*100, "%"))

rf_model = randomForest(y ~ ., data=train_data2, ntree = 300)
output_rf = predict(rf_model, x_test2)
prediction_rf = data.frame(y = round(output_rf))
print(paste("Accuracy of random forest model: ",(sum(prediction_rf == y_test)/len_test)*100, "%"))
as.numeric(output_rf>mean(output_rf))
print(paste("Accuracy of random forest model: ",(sum(as.numeric(output_rf>mean(output_rf)) == y_test)/len_test)*100, "%"))

h2o.init()
dl_data <- as.h2o(train_data2)
dl_model <- h2o.deeplearning(x = 1:50, y = 51,
                             training_frame = dl_data,
                             hidden = c(100,100,100),
                             epochs = 300)

prediction_dl = h2o.predict(dl_model, as.h2o(x_test2))
prediction_dl = round(as.data.frame(prediction_dl))
print(paste("Accuracy of deep learning model: ",(sum(prediction_dl == y_test)/len_test)*100, "%"))
