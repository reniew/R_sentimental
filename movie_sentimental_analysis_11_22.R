#data = read.delim('./data/labeledTrainData.tsv', header = 1, quote = "",as.is=T)
#attach(data) # id, review, sentiment

# import library

install.packages("text2vec")
install.packages("tm")
install.packages("softmaxreg")
library("text2vec")
library("tm")
library("softmaxreg")

# load data 

data("movie_review")
review = movie_review$review
label = movie_review$sentiment
head(review)

# preprocessing data

review = gsub("<br />", "", review)
review = gsub("[^a-zA-Z /s]", "", review) # remove all character without alphabet&space
review = gsub("[^a-zA-Z]", " ", review) # remove special character beside space
review = tolower(review)
head(review)

# tokenizing string to list of word

tokenized_review = word_tokenizer(review)
head(tokenized_review)

# set stopwords

stopword = stopwords()

# remove stopword

'%nin%' <- Negate('%in%')
tokenized_review = lapply(tokenized_review, function(x){
  x[x %nin% stopword]
})

# joining review to string

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

