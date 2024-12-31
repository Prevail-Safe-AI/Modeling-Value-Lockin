import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Make sure these downloads have been done once in your environment:
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def simplify_text(text: str) -> str:
    """
    Reduces a string (English phrase) to its simplest form by:
    1. Lowercasing
    2. Removing punctuation
    3. Tokenizing
    4. Removing English stopwords
    5. Lemmatizing
    """

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove punctuation
    #    You can also strip out other characters (e.g., digits) as needed.
    #    string.punctuation is typically: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Tokenize the text
    tokens = word_tokenize(text)

    # 4. Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(filtered_tokens)

    # 5. Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in pos_tag(word_tokenize(text))]

    # Join tokens back to a single string (optional)
    simplified_text = " ".join(lemmatized_tokens)
    return simplified_text

# Example usage
if __name__ == "__main__":
    example_sentence = "I am craving for blood oranges"
    print("Original:", example_sentence)
    print("Simplified:", simplify_text(example_sentence))  # crave blood orange