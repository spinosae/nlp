from __future__ import division
import csv, nltk, random, string, pickle, re, itertools, sklearn
from operator import add
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import tagger.CMUTweetTagger as tagger

# nltk.download("all")
TAG = "java -XX:ParallelGCThreads=2 -Xmx500m -jar tagger/ark-tweet-nlp-0.3.2.jar --model tagger/model.penn"
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def not_punct(token):
    if (re.match("[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]", token)):
        return False
    else:
        return True


def not_garbled(token):
    if (re.match("[?]{2,}", token)):
        return False
    else:
        return True


def qualified(token, keep_punc):
    return not_garbled(token) and keep_punc or not_punct(token)


def read_file(in_file):
    with open(in_file, 'r') as file:
        reader = csv.reader(file)
        return [(line[0].decode('ascii', 'ignore'), line[1]) for line in reader]


def tag_doc(document):
    tweets, labels = zip(*document)
    tweets_tagged = tagger.runtagger_parse(tweets, TAG)
    return zip(tweets_tagged, labels)


def create_pickles(in_file, out_file):
    print(in_file + " " + out_file)
    document = read_file(in_file)
    tag = tag_doc(document)
    with open(out_file, "wb") as out_stream:
        pickle.dump(tag, out_stream)
    return tag


def load_pickles(in_files):
    data = []
    for in_file in in_files:
        print(in_file)
        with open(in_file, "rb") as in_data:
            data.append(pickle.load(in_data))
    return data


def process_token(token, stem, lemma):
    if (stem):
        token = stemmer.stem(token)
    if (lemma):
        token = lemmatizer.lemmatize(token)
    return token


def clean(tagged_doc, stem, lemma, keep_punc, keep_tag):
    tweets, labels = map(list, zip(*tagged_doc))
    tweets = [[(process_token(token, stem, lemma), tag) for (token, tag, prob) in tweet if
                 qualified(token, keep_punc)] for tweet in tweets]
    if(not keep_tag):
        tweets = de_tag(tweets)
    return zip(tweets, labels)


def de_tag(tweets):
    return [[token for (token, tag) in tweet] for tweet in tweets]


def de_label(docs_features):
    attrs, labels = map(list, zip(*docs_features))
    return (attrs,labels)

def assemble_tokens(tokens):
    return ' '.join(tokens.keys())

def split_docs(docs, indices):
    train_set = reduce(lambda x, y: x+y, [docs[i] for i in indices['train']])
    test_set = reduce(lambda x, y: x+y, [docs[i] for i in indices['test']])
    return (train_set, test_set)


def test(a,b,c):
    print(str(len(a)) + " " +str(len(b)) +" "+ str(len(c)))


def featurize(docs, features):
    return [({feature: feature in tokens for feature in features}, label) for (tokens, label) in docs]


def gen_index(num, select):
    num_l = range(num)
    trains = [list(x) for x in itertools.combinations(num_l,select)]
    tests = [[x for x in num_l if not x in train] for train in trains]
    combined = zip(trains, tests)
    return [dict(zip(['train','test'], d)) for d in combined]


def run_test(params, indice):
    train_set, test_set = map(lambda x:clean(x, *params.values()), split_docs(docs, indice))
    all_tokens = [token for (tweet, label) in train_set for token in tweet]
    # all_tokens = set(token for (tweet, label) in train_set for token in tweet)
    features = [x for (x, freq) in nltk.FreqDist(all_tokens).most_common() if
                not x[0] in nltk.corpus.stopwords.words('english')]
    train_set_features, test_set_features = map(lambda x:featurize(x,features), [train_set,test_set])
    fit = nltk.NaiveBayesClassifier.train(train_set_features)
    fit.show_most_informative_features()
    attrs, labels = de_label(test_set_features)
    predicts = fit.classify_many(attrs)
    session_info = '##clean parameters: '+str(params) +'\n'
    session_info += '##indices: '+str(indice) + '\n\n'
    print(session_info)
    errors = ""
    for i in range(len(predicts)):
        if predicts[i] != labels[i]:
            error_tweet = test_set[i][0]
            print(error_tweet)
            if params["pos-tag"]:
                error_tweet = [word for (word, tag) in error_tweet]
            errors += 'text line {!s}: {}\nguess: {} label: {}\n'.format(i, " ".join(error_tweet), predicts[i], labels[i])
    errors += '\n'
    report = sklearn.metrics.classification_report(labels, predicts)
    values = re.findall("(?<=\W)\d*\.?\d+(?=\W)|/", report)
    values = [[values[i*5 + j] for j in range(5)] for i in range(3)]
    values = map(float, values[2][1:4])
    report = session_info  + report + '\n\n'
    errors = session_info  + errors + '\n\n'
    return (report, values, errors)

def process_result(results):
    report, values, errors = map(list, zip(*results))
    report = reduce(lambda x, y:x+y, report)
    errors = reduce(lambda x, y:x+y, errors)
    values = reduce(lambda x, y:map(add, x, y), values)
    values = [value/len(values) for value in values]
    avg = ['{0:.2f}'.format(x) for x in values]
    report += '\navg / total       {avg[0]}      {avg[1]}      {avg[2]}       \n'.format(avg=avg)
    return (report, errors, values)

def auto_run(params):
    out = '_'.join([key for key in params.keys() if params[key]])
    results = map(lambda x: run_test(params, x), indices)
    report, errors, values = process_result(results)
    with open(out + '_report.txt', 'w') as rep:
        rep.write(report)
    with open(out + '_errors.txt', 'w') as err:
        err.write(errors)

csv_files = ["set" + str(index) + ".csv" for index in range(1, 4)]
pickle_files = ["set" + str(index) + "_tag.pickle" for index in range(1, 4)]
docs = load_pickles(pickle_files)
indices = gen_index(len(docs), 2)
keys = ['stemming', 'lemmatization', 'punctuation', 'pos-tag']

values = [[True, False, False, False],
          [False, True, False, False],
          [False, False, False, True],
          [False, False, False, False],
          [False, False, True, False],
          [False, False, False, False]
          ]
params = [dict(zip(keys, value)) for value in values]
map(auto_run, params)

