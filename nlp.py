import csv, nltk, random, string, pickle, re, itertools
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import tagger.CMUTweetTagger as tagger


TAG = "java -XX:ParallelGCThreads=2 -Xmx500m -jar tagger/ark-tweet-nlp-0.3.2.jar --model tagger/model.penn"
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def not_punct(token):
    if(re.match("[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]", token)):
        return False
    else:
        return True


def not_garbled(token):
    if(re.match("[?]{2,}", token)):
        return False
    else:
        return True


def qualified(token, keep_punc):
     return not_garbled(token) and keep_punc or not_punct(token)


def read_file(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        return [(line[0].decode('ascii','ignore'), line[1]) for line in reader if line[1]]


def tag_doc(document):
    sents, labels = zip(*document)
    tag_tokens = tagger.runtagger_parse(sents, TAG)
    return zip(tag_tokens, labels)


def proc_token(token, stem, lemma):
    if(stem):
        token = stemmer.stem(token)
    if(lemma):
        token = lemmatizer.lemmatize(token)
    return token


def clean(tag, stem, lemma, keep_punc):
    return [(set((proc_token(token, stem, lemma), tag) for (token, tag, prob) in tokens if qualified(token, keep_punc)), label) for (tokens, label) in tag]


def de_tag(tag):
    return [(set(token for (token, tag) in tokens), label) for (tokens, label) in tag]


def tokenize(dataset):
    return [([stemmer.stem(word) for word in nltk.word_tokenize(sent)], label) for (sent, label) in dataset]


files = [("set"+str(index)+".csv", "set"+str(index)+"_tag.pickle") for index in range(1,4)]
data = []
for inp, oup in files:
    print(inp + " " +oup)
    document = read_file(inp)
    tag = tag_doc(document)
    # cleaned = clean(tag, True, False, True)
    data.append(tag)
    # with open(oup,"wb") as outo:
    #     pickle.dump(tag, outo)

train_set = data[0] + data[2]
train_set = clean(train_set, True, False, False)
test_set = data[1]
test_set = clean(test_set, True, False, False)

all_tokens = [token for (tweet, label) in train_set for token in tweet]
features = [x for (x,freq) in nltk.FreqDist(all_tokens).most_common() if not x[0] in nltk.corpus.stopwords.words('english')]
train_set_features = [({feature: feature in tokens for feature in features }, label) for (tokens, label) in train_set]
test_set_features = [({feature: feature in tokens for feature in features }, label) for (tokens, label) in test_set]
fit = nltk.NaiveBayesClassifier.train(train_set_features)
fit.show_most_informative_features()
attrs, labels = map(list, zip(*test_set_features))
predicts = fit.classify_many(attrs)

print(nltk.classify.accuracy(fit,test_set_features))

