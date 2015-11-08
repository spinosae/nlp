# Sentiment analysis tool for tweets

This project was dedicated to polarity detection of Twitter post regarding Apple and its products.
## Dependency
The project depends on a few open source projects.
- [Twitter NLP][tnlp] for tokenization and POS-tagging
- [Twitter NLP Python Wrapper][tnlpwp] for integration of Twitter NLP on Python
- [NLTK][nltk]  for text classification
- [scikit-learn][sklearn] for performance measurement
- various other modules on Python
## Installation
Twitter NLP and Twitter NLP Wrapper were provided in *tagger* folder and require no further installation. However, NLTK and Scikit-Learn need to be installed before the program would run. Installations were tested on Fedora 23 and may differ for other platforms.

*Note: Twitter NLP Wrapper doesn't run on Windows, tagged data (*.pickle) were provided in data folder. However, if you want to run POS-tag again, run it on Linux.*

#### Install scikit-learn
```bash
$ pip install scikit-learn
```
Above installation may fail if pre-requisites such as NumPy and SciPy packages of scikit-learn were not installed. Please install them with your package manager. For Fedora, run:
```bash
$ sudo dnf -y install gcc gcc-c++ numpy python-devel scipy
```
#### Install NTLK
```bash
$ pip install nltk
```
After installing NLTK, you need to download corporas etc. A lazy way is to download all resources for NLTK.
```
>>> import nltk
>>> nltk.download('all')
```
## Run
With data in place and installed dependencies, simply
```
$ python nlp.py
```
would perform classification automatically. Output is saved into output folder. There are 3 types of output.
1. *_report.txt contains performance metrics including precision, recall and f-measure for each fold of cross-validation as well as averaged scores.
2. *_errors.txt contains records where model prediction doesn't agree with actual label
3. table.csv is an aggregated table of results.
File name is deliberated, it reveals techniques and features enabled during the learning process. For example:
**punctuation_lemmatization_errors.txt** means punctuations were kept and lemmatization was used.

   [tnlp]: http://www.ark.cs.cmu.edu/TweetNLP/ "Twitter NLP"
   [tnlpwp]: https://github.com/ianozsvald/ark-tweet-nlp-python "Twitter NLP Python wrapper"
   [nltk]: http://www.nltk.org/ "NLTK"
   [sklearn]: http://scikit-learn.org/ "Scikit-Learn"
