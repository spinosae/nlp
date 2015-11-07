import csv, nltk, random, string, pickle
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import tagger.CMUTweetTagger as tagger

TAG = "java -XX:ParallelGCThreads=2 -Xmx500m -jar tagger/ark-tweet-nlp-0.3.2.jar --model tagger/model.penn"
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def read_file(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        return [(line[0], line[1]) for line in reader if line[1]]

def tag_file(filepath):
    document = read_file(filepath)
    sents, labels = zip(*document)
    tag_tokens = tagger.runtagger_parse(sents, TAG)
    return zip(tag_tokens, labels)

def legitimate(token):

def clean(tag):
    return [([(token, tag) for (token, tag, prob) in tokens if legitimate(token)], label) for (tokens, label) in tag]

def tokenize(dataset):
    stemmer=SnowballStemmer('english')
    return [([stemmer.stem(word) for word in nltk.word_tokenize(sent)], label) for (sent, label) in dataset]

tweets = read_file('item_shiffle2.csv')
tweets_tokens = tokenize(tweets)
all_tokens = [token for (tweet, label) in tweets_tokens for token in tweet]
features = [x for (x,freq) in nltk.FreqDist(all_tokens).most_common() if not x in nltk.corpus.stopwords.words('english')]
feature_set = [({feature: feature in tokens for feature in features }, label) for (tokens, label) in tweets_tokens]
shuffled = random.shuffle(feature_set)
pickle.dump(feature_set, open( "toJiachun.p", "wb" ) )
train_size = len(feature_set)*2/3
train = feature_set[0:train_size]
test = feature_set[train_size:]
fit = nltk.NaiveBayesClassifier.train(train)
print(nltk.classify.accuracy(fit,test))
