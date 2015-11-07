import pickle
import nltk
from nltk.stem import *
import sys
import string
reload(sys)
sys.setdefaultencoding("utf-8")
tag = pickle.load(open("set1_tag.p", "rb"))


all_tokens_tag = [([(token, tag) for (token, tag, prob) in tokens], label) for (tokens, label) in tag]
#print stripped
#all_tokens=[token for (token, tag)]
stemmer=SnowballStemmer('english')
all_tokens = [stemmer.stem(token) for (tweet, label) in all_tokens_tag for token in tweet]
features = [x for (x,freq) in nltk.FreqDist(all_tokens).most_common() if not unicode(x[0]) in nltk.corpus.stopwords.words("english")]
print features
feature_set = [({feature: feature in tokens for feature in features }, label) for (tokens, label) in all_tokens_tag]
train_size = len(feature_set)*2/3
train = feature_set[train_size:]
test = feature_set[:train_size]
fit = nltk.NaiveBayesClassifier.train(train)
print(nltk.classify.accuracy(fit,test))
#print features
