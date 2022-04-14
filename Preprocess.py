from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet
from textblob import TextBlob
import nltk
import re


def preprocess(dataset):
    for i in dataset.index:
        # remove uppercase letters
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].lower() if type(dataset.loc[i, 'tweet']) == str else \
            dataset.loc[i, 'tweet']

        # convert emojis and emoticons to text
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].apply(emojisToWords)
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].apply(emoticonsToWords)

        # remove URLs
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].replace(r'^https?:\/\/.*[\r\n]*', '')

        # remove numbers
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].replace(r'\d+', '')

        # remove \n, \u0111
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].replace('\n', '')
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].replace('\u0111', '')

        # remove punctuation
        dataset['tweet'] = dataset['tweet'].str.replace('[^A-Za-z0-9-\s]+', '') if type(
            dataset.loc[i, 'tweet']) == str else dataset.loc[i, 'tweet']

        # remove stopwords
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].apply(removeStopwords)

        # removes most common words
        # cnt = findMostCommonWords(dataset)
        # dataset.loc[i,'tweet'] = dataset.loc[i,'tweet'].apply(removeCommonWords)

        # Rare word removal

        # spelling correction
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].apply(lambda x: str(TextBlob(x).correct()))

        # remove whitespaces
        # dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].str.strip()

        # tokenization

        # lemmatization
        dataset.loc[i, 'tweet'] = dataset.loc[i, 'tweet'].apply(lemmatize_words)

    return dataset


def emojisToWords(text):
    for emoji in UNICODE_EMOJI:
        text = text.replace(emoji, "_".join(UNICODE_EMOJI[emoji].replace(",", "").replace(":", "").split()))
        return text


def emoticonsToWords(text):
    for emoticon in EMOTICONS_EMO:
        text = re.sub(u'(' + emoticon + ')', "_".join(EMOTICONS_EMO[emoticon].replace(",", "").split()), text)
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
    return " ".join(
        [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
