import spacy
import emoji
import string
from typing import List, Text
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sentiment.config import logger

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


class Augmentation(object):

    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    @staticmethod
    def clean(text: Text) -> Text:
        return REPLACE_WITH_SPACE.sub(" ", REPLACE_NO_SPACE.sub("", text.lower()))

    @staticmethod
    def strip_all_entities(tweet: Text) -> Text:
        entity_prefixes = ['@']
        words = []
        text = None

        for separator in string.punctuation:
            if separator not in entity_prefixes:
                text = tweet.replace(separator, ' ')

        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)

        return ' '.join(words)

    @staticmethod
    def remove_markup(tweet: Text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', tweet)

        return cleantext

    @staticmethod
    def remove_emoji(tweet: Text):
        allchars = [str for str in tweet.encode('utf-8').decode()]
        EMOJIS = []

        for key, value in emoji.UNICODE_EMOJI.items():
            EMOJIS.extend([
                emoji for emoji, _ in value.items()
            ])

        emoji_list = [c for c in allchars if c in EMOJIS]
        return ' '.join(
            [
                str for str in tweet.encode('utf-8').decode().split()
                if not any(i in str for i in emoji_list)
            ]
        )

    @staticmethod
    def remove_url(tweet: Text) -> Text:
        return re.sub(r'http\S+', '', tweet)

    @staticmethod
    def remove_symbols(tweet: Text) -> Text:
        return re.sub('[^ A-Za-z0-9]+', '', tweet).lower()

    @staticmethod
    def spell_correction(tweet: Text) -> Text:

        spell = SpellChecker()

        return " ".join(
            spell.correction(word) for word in tweet.split(" ")
        )

    @staticmethod
    def tokenize(text: str):

        try:
            NLP = spacy.load("en_core_web_sm")
            document = NLP(text)

        except Exception as _exc:
            logger.exception(f"Error while processing text. Reason: {_exc}")
            return None

        return document

    @staticmethod
    def remove_stopwords(tokens):
        return [
            token for token in tokens if not token.is_stop
        ]

    @staticmethod
    def stemming(words: List[str]) -> List[str]:
        stemmer = PorterStemmer()
        return [
            stemmer.stem(word) for word in words if word
        ]

    @staticmethod
    def lemmatization(words: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        return [
            lemmatizer.lemmatize(word) for word in words
        ]

    def augment(self, text):
        logger.debug(f"Original text: {text}")

        cleaned_text = self.clean(text)
        logger.debug(f"Cleaned text: {cleaned_text}")

        no_entities_text = self.strip_all_entities(cleaned_text)
        logger.debug(f"No @ in text: {no_entities_text}")

        no_markup_text = self.remove_markup(no_entities_text)
        logger.debug(f"No markup text: {no_markup_text}")

        no_emoji_text = self.remove_emoji(no_markup_text)
        logger.debug(f"No emoji text: {no_emoji_text}")

        no_url_text = self.remove_url(no_emoji_text)
        logger.debug(f"No url text: {no_url_text}")

        no_symbols_text = self.remove_symbols(no_url_text)
        logger.debug(f"No symbols text: {no_symbols_text}")

        spell_corrected_text = self.spell_correction(no_symbols_text)
        logger.debug(f"Spell corrected text: {spell_corrected_text}")

        stem = self.stemming(spell_corrected_text.split(" "))
        logger.debug(f"Stemming text: {stem}")

        lem = self.lemmatization(stem)
        logger.debug(f"Lemmatizing text: {lem}")

        tokenized_text = self.tokenize(" ".join(_tweet for _tweet in lem))
        logger.debug(f"Tokenized text: {tokenized_text}")

        no_stopwords_text = self.remove_stopwords(tokenized_text)
        logger.debug(f"No stopwords text: {no_stopwords_text}")

        return " ".join(t.text for t in no_stopwords_text)


