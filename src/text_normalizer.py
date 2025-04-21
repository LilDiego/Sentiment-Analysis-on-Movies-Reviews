import re
import subprocess
import unicodedata
from typing import List, Optional

import nltk
import spacy
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from gensim.utils import deaccent

from src.contractions import CONTRACTION_MAP

# Download the models used
nltk.download("stopwords")
nltk.download("punkt")
subprocess.run(["spacy", "download", "en_core_web_sm"])

# Load NLP models
tokenizer = ToktokTokenizer()
nlp = spacy.load("en_core_web_sm")
stopword_list = nltk.corpus.stopwords.words("english")


def remove_html_tags(text: str) -> str:
    """
    Remove html tags from text like <br/> , etc. You can use BeautifulSoup for this.

    Args:
        text : str
            Input string.

    Return:
        str
            Output string.
    """
    # TODO
    # Parse the text as HTML
    soup = bs(text, "html.parser")
    
    # Extract and return the plain text without HTML tags
    return soup.get_text()

def stem_text(text: str) -> str:
    """
    Stem input string.
    (*) Hint:
        - Use `nltk.porter.PorterStemmer` to pass this test.
        - Use `nltk.tokenize.word_tokenize` for tokenizing the sentence.

    Args:
        text : str
            Input string.

    Return:
        str
            Output string.
    """
    # TODO
    # Tokenize the text
    tokens = word_tokenize(text)

    # Create the stemmer object
    stemmer = nltk.stem.PorterStemmer()

    # Apply stemming to each token
    stemmed_words = [stemmer.stem(word) for word in tokens]

    # Join the stemmed words into a single string
    return " ".join(stemmed_words)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize input string, tokenizing first and extracting lemma from each text after.
    (*) Hint: Use `nlp` (spacy model) defined in the beginning for tokenizing
    and getting lemmas.

    Args:
        text : str
            Input string.

    Return:
        str
            Output string.
    """
    # TODO
    # Process the text with Spacy Model
    doc =  nlp(text)

    lemmas = [token.lemma_ for token in doc]

    #join lemmas
    return " ".join(lemmas)



def remove_accented_chars(text: str) -> str:
    """
    Remove accents from input string.

    Args:
        text : str
            Input string.

    Return:
        str
            Output string.
    """
    # TODO
    return deaccent(text)


def remove_special_chars(text: str, remove_digits: Optional[bool] = False) -> str:
    """
    Remove non-alphanumeric characters from input string.

    Args:
        text : str
            Input string.
        remove_digits : bool
            Remove digits.

    Return:
        str
            Output string.
    """
    # TODO
    # Build the regular expression depending on `remove_digits`
    pattern = r"[^a-zA-Z0-9\s]" if not remove_digits else r"[^a-zA-Z\s]"

    # Use re.sub to replace unwanted characters with an empty string
    cleaned_text = re.sub(pattern, "", text)

    # Return the cleaned text
    return cleaned_text


def remove_stopwords(
    text: str,
    is_lower_case: Optional[bool] = False,
    stopwords: Optional[List[str]] = stopword_list,
) -> str:
    """
    Remove stop words using list from input string.
    (*) Hint: Use tokenizer (ToktokTokenizer) defined in the beginning for
    tokenization.

    Args:
        text : str
            Input string.
        is_lower_case : bool
            Flag for lowercase.
        stopwords : List[str]
            Stopword list.

    Return:
        str
            Output string.
    """
    # TODO
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)

    # Normalize tokens and stopwords to lowercase for comparison
    filtered_tokens = [
        token for token in tokens 
        if token.lower() not in stopwords
    ]

    # Return the cleaned string
    return " ".join(filtered_tokens)


def remove_extra_new_lines(text: str) -> str:
    """
    Remove extra new lines or tab from input string.

    Args:
        text : str
            Input string.

    Return:
        str
            Output string.
    """
    # TODO
    # Replace newlines and tabs with a single space
    cleaned_text = text.replace("\n", " ").replace("\t", " ")

    # Remove any extra spaces that may have been generated
    return " ".join(cleaned_text.split())


def remove_extra_whitespace(text: str) -> str:
    """
    Remove any whitespace from input string.

    Args:
        text : str
            Input string.

    Return:
        str
            Output string.
    """
    # TODO
    # Replace multiple whitespace characters with a single space
    cleaned_text = " ".join(text.split())
    
    # Return the cleaned text
    return cleaned_text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP) -> str:
    """
    Expand english contractions on input string.

    Args:
        text : str
            Input string.
    Return:
        str
            Output string.
    """
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    text = re.sub("'", "", expanded_text)

    return text


def normalize_corpus(
    corpus: List[str],
    html_stripping: Optional[bool] = True,
    contraction_expansion: Optional[bool] = True,
    accented_char_removal: Optional[bool] = True,
    text_lower_case: Optional[bool] = True,
    text_stemming: Optional[bool] = False,
    text_lemmatization: Optional[bool] = False,
    special_char_removal: Optional[bool] = True,
    remove_digits: Optional[bool] = True,
    stopword_removal: Optional[bool] = True,
    stopwords: Optional[List[str]] = stopword_list,
) -> List[str]:
    """
    Normalize list of strings (corpus)

    Args:
        corpus : List[str]
            Text corpus.
        html_stripping : bool
            Html stripping,
        contraction_expansion : bool
            Contraction expansion,
        accented_char_removal : bool
            accented char removal,
        text_lower_case : bool
            Text lower case,
        text_stemming : bool
            Text stemming,
        text_lemmatization : bool
            Text lemmatization,
        special_char_removal : bool
            Special char removal,
        remove_digits : bool
            Remove digits,
        stopword_removal : bool
            Stopword removal,
        stopwords : List[str]
            Stopword list.

    Return:
        List[str]
            Normalized corpus.
    """

    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)

        # Remove extra newlines
        doc = remove_extra_new_lines(doc)

        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # Expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)

        # Remove special chars and\or digits
        if special_char_removal:
            doc = remove_special_chars(doc, remove_digits=remove_digits)

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

        # Lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc, is_lower_case=text_lower_case, stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()

        normalized_corpus.append(doc)

    return normalized_corpus
