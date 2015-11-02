import csv, nltk, random
from nltk.stem import *
import string

def read_file(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        return [(str(line[0]).translate(None, string.punctuation).decode("utf8"), line[1]) for line in reader if line[1] in ['1','3']]


def tokenize(dataset):
    stemmer=SnowballStemmer('english')
    return [([stemmer.stem(word) for word in nltk.word_tokenize(sent)], label) for (sent, label) in dataset]

tweets = read_file('item_shiffle2.csv')
tweets_tokens = tokenize(tweets)
all_tokens = [token for (tweet, label) in tweets_tokens for token in tweet]
features = [x for (x,freq) in nltk.FreqDist(all_tokens).most_common() if not x in nltk.corpus.stopwords.words('english')]
feature_set = [({feature: feature in tokens for feature in features }, label) for (tokens, label) in tweets_tokens]
shuffled = random.shuffle(feature_set)
train_size = len(feature_set)*2/3
train = feature_set[0:train_size]
test = feature_set[train_size:]
fit = nltk.NaiveBayesClassifier.train(train)
print(nltk.classify.accuracy(fit,test))
