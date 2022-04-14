from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet
from textblob import TextBlob
from emot import emo_unicode
import nltk
import re


def preprocess(dataset):
    # remove uppercase letters
    dataset['tweet'] = dataset['tweet'].apply(lambda s: s.lower() if type(s) == str else s)

    # convert emojis and emoticons to text
    dataset['tweet'] = dataset['tweet'].apply(emojisToWords)
    #dataset['tweet'] = dataset['tweet'].apply(emoticonsToWords)

    # remove URLs
    dataset['tweet'] = dataset['tweet'].replace(to_replace=r'^https?:\/\/.*[\r\n]*', value='', regex=True)

    # remove numbers
    dataset['tweet'] = dataset['tweet'].replace(to_replace=r'\d+', value='', regex=True)

    # remove \n, \u0111
    dataset['tweet'] = dataset['tweet'].replace(to_replace='\n', value='', regex=True)
    dataset['tweet'] = dataset['tweet'].replace(to_replace='\u0111', value='', regex=True)

    # remove punctuation
    dataset['tweet'] = dataset['tweet'].str.replace('[^\w\s]', '')  # replace(r'^https?:\/\/.*[\r?\n\t@]*', '')

    # remove stopwords
    dataset['tweet'] = dataset['tweet'].apply(removeStopwords)

    # removes most common words
    cnt = findMostCommonWords(dataset)
    dataset['tweet'] = dataset["tweet"].apply(removeCommonWords)

    # Rare word removal

    # spelling correction
    dataset['tweet'].apply(lambda x: str(TextBlob(x).correct()))

    # remove whitespaces
    # dataset['tweet'].str.strip()

    # tokenization

    # Lemmatization
    dataset['tweet'] = dataset['tweet'].apply(lemmatize_words)

    return dataset


def emojisToWords(text):
    for emot in emo_unicode.EMOJI_UNICODE:
        text = text.replace(emot, "_".join(emo_unicode.EMOJI_UNICODE[emot].replace(",", "").replace(":", "").split()))
        return text


def emoticonsToWords(text):
    for emot in emo_unicode.EMOTICONS_EMO:
        text = re.sub(u'(' + emot + ')', "_".join(emo_unicode.EMOTICONS_EMO[emot].replace(",", "").split()), text)
        return text


def removeStopwords(text):
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if
                     word not in STOPWORDS])  # Applying the stopwords to 'text_punct' and store into 'text_stop'


def findMostCommonWords(dataset):
    cnt = Counter()
    for text in dataset["text_stop"].values:
        for word in text.split():
            cnt[word] += 1

    return cnt


def removeCommonWords(text, toRemove, cnt):
    return " ".join([word for word in str(text).split() if word not
                     in cnt.most_common(toRemove)])

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])