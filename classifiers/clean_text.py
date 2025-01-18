import re
import nltk
import string
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize necessary NLTK components
stopwords_set = set(stopwords.words('english'))
custom_words = {"accessible", "accessibility", "handicap", "handrail", "blind", "deaf", "wheelchair"}
stopwords_set.update(custom_words)

lemmatizer = WordNetLemmatizer()

# Define a regex pattern to remove emojis
emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def remove_html_tags(text):
    """Remove HTML tags from the text."""
    return BeautifulSoup(text, "html.parser").get_text()

def remove_non_ascii(text):
    """Remove non-ASCII characters from the text."""
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

def remove_emoji(text):
    """Remove emojis from the text."""
    return emoji_pattern.sub(r'', text)

def normalize_whitespace(text):
    """Normalize whitespace in the text."""
    return re.sub(r'\s+', ' ', text).strip()

def replace_fancy_quotes(text):
    """Replace fancy quotes with standard quotes."""
    return re.sub(r"[‘’“”]", "'", text)

def remove_punctuations(text):
    """Remove punctuations from the text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_digits(text):
    """Remove digits from the text."""
    return ''.join(char for char in text if not char.isdigit())

def remove_urls(text):
    """Remove URLs from the text."""
    return re.sub(r'http\S+', '', text)

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return ' '.join(word for word in text.split() if word not in stopwords_set)

def nltk_to_wordnet_tag(nltk_tag):
    """Convert NLTK POS tag to WordNet POS tag."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    return None

def lemmatize_text(text):
    """Lemmatize the text."""
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    wn_tagged = [(word, nltk_to_wordnet_tag(tag)) for word, tag in nltk_tagged]
    lemmatized_words = [lemmatizer.lemmatize(word, tag) if tag else word for word, tag in wn_tagged]
    return " ".join(lemmatized_words)

def clean_text(text):
    """Clean the text by applying all the preprocessing steps."""
    text = remove_html_tags(text)
    text = remove_non_ascii(text)
    text = remove_emoji(text)
    text = normalize_whitespace(text)
    text = replace_fancy_quotes(text)
    text = remove_punctuations(text)
    text = remove_digits(text)
    text = remove_urls(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = text.lower()
    return text