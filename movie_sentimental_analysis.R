#data = read.delim('./data/labeledTrainData.tsv', header = 1, quote = "",as.is=T)
#attach(data) # id, review, sentiment

install.packages("text2vec")
install.packages("tm")
install.packages("softmaxreg")
library("text2vec")
library("tm")
library("softmaxreg")


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

data("word2vec")
x = list()

for (token in temp[[1]]) {
  wordEmbed(token, word2vec)
}



for (i in range(length(dim(a)[1]))) {
  
}