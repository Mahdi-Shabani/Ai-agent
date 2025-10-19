import nltk
import spacy
import re
import string
import io, contextlib
from nltk.corpus import wordnet as wn
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer
from spacy.matcher import Matcher
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Download essential NLTK datasets: tokenizers, stopwords, WordNet, etc.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('gutenberg', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Download and load the small English model for spaCy
# py -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download as spacy_download
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# our example in this work
short_text = "Natural Language Processing is fun and educational."
long_text = 'Dr. Smith, traveled to Washington, D.C. on Jan. 5th for a cutting-edge NLP conference. During his keynote, he explained that advancements in tokenization techniques—particularly those implemented in NLTK and spaCy (e.g., handling abbreviations like "Dr." and "e.g." seamlessly)—are transforming text analysis.'
text_emails = "Contact us at admin.support_34@example.com or sales-dep@company.org for inquiries."


# Corpora and Lexical Resources
# Corpora and Lexical Resources are essential for understanding language structure and semantics.
# They provide extensive collections of texts (corpora) and structured word relationships (lexical databases)
# that support tasks like language modeling, semantic analysis, and more.

# Retrieve synsets for the word 'computer'
synsets = wn.synsets('great')
print("WordNet Synsets for 'computer':", synsets)
print('-----------------Gutenberg Files------------------')
# List available files in the Gutenberg corpus
print("Gutenberg Files:", gutenberg.fileids())

# spaCy Lexical Resources
# While spaCy does not include separate corpora like NLTK, 
# its language models (e.g., en_core_web_sm) incorporate rich lexical data,
#  including vocabulary, part-of-speech tags, and dependency structures.

print('-----------------spaCy Lexical Resources------------------')
doc = nlp(short_text)

# Print token, lemma, POS tag, and dependency relation for each token
for token in doc:
    print(f"Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}, Dep: {token.dep_}")


# Tokenization Techniques
# Tokenization is the process of splitting text into smaller units such as words or sentences,
#  which simplifies subsequent text analysis and processing.
print('-----------------Tokenization Techniques------------------')

# NLTK Tokenization
# Word Tokenization with NLTK
long_text = 'Dr. Smith, traveled to Washington, D.C. on Jan. 5th for a cutting-edge NLP conference! During his keynote, he explained that advancements in tokenization techniques—particularly those implemented in NLTK and spaCy (e.g., handling abbreviations like "Dr." and "e.g." seamlessly)—are transforming text analysis.'

words = nltk.word_tokenize(long_text)
print("Word Tokens:", words)

# Sentence Tokenization with NLTK:
sentences = nltk.sent_tokenize(long_text)
print("Sentence Tokens:", sentences)
print(len(sentences))

print('-----------------spaCy Tokenization------------------')
# spaCy Tokenization
# Word Tokenization with spaCy:

doc = nlp(short_text)
words = [token.text for token in doc]
print("Word Tokens:", words)

# Sentence Tokenization with spaCy
sentences = [sent.text for sent in doc.sents]
print("Sentence Tokens:", sentences)

print('-----------------Regex for Pattern Matching------------------')
# 4.Regex for Pattern Matching
# Regex for pattern matching is a powerful technique to extract or filter specific text patterns.
# With NLTK, you can leverage its RegexpTokenizer to tokenize text based on custom regex patterns,
# while spaCy’s Matcher enables regex-like matching within its robust linguistic framework.
text = "Hello world! This is an example: email@example.com, phone: 123-456-7890."

# Define a tokenizer that captures alphanumeric words
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)
print("NLTK Regex Tokens:", tokens)

print('-----------------spaCy Regex Example------------------')
# spaCy Regex Example
# Load the small English model
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define a regex-based pattern: match tokens that start with a capital letter
pattern = [{"TEXT": {"REGEX": "^[A-Z][a-z]+"}}]
matcher.add("CAPITAL_PATTERN", [pattern])

text = "Hello world! This is an Example sentence."
doc = nlp(text)

# Apply the matcher and print matched tokens
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print("Matched token:", span.text)

print('-----------------Example 1: Email Detection------------------')
text_emails = "Contact us at admin.support_34@example.com or sales-dep@company.org for inquiries."

# Example 1: Email Detection
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # Regex for email matching

emails = re.findall(email_pattern, text_emails)
print("Detected Emails:", emails)

print('-----------------Stopwords Filtering------------------')
# Stopwords Filtering
# Stopwords filtering involves removing commonly used words (e.g., "the", "is", "at")
# that often add little semantic value to text analysis.
# This process helps focus on more meaningful terms. 
# NLTK offers a comprehensive list of stopwords,
# whereas spaCy incorporates an internal attribute for each token to determine if it is a stopword.

# NLTK Stopwords Filtering

text = "This is a simple example demonstrating stopword removal in natural language processing."
words = nltk.word_tokenize(text)
filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
print("Filtered Words:", filtered_words)

print('-----------------spaCy Stopwords Filtering------------------')
# spaCy Stopwords Filtering
doc = nlp(long_text)
filtered_tokens = [token.text for token in doc if not token.is_stop]
print("Filtered Tokens:", filtered_tokens)

print('-----------------Stemming Methods------------------')
# Stemming Methods
# Stemming reduces words to their root forms by removing affixes, 
# thus simplifying text for analysis and retrieval tasks. 
# NLTK offers robust stemming algorithms like Porter, 
# Lancaster, and Snowball. While spaCy focuses on lemmatization for morphological normalization, 
# you can integrate NLTK's stemmers with spaCy's tokenization if stemming is required

# NLTK Stemming Example
# Initialize the PorterStemmer
ps = PorterStemmer()
sn = SnowballStemmer(language="english")
text = "running runner easily study hardly activity ran"
words = nltk.word_tokenize(text)
stems = [ps.stem(word) for word in words]

sn_stems = [ps.stem(word) for word in words]
print("NLTK Stemmed Words:", stems)
print("NLTK Snowball Words:", sn_stems)

print('-----------------Integrating NLTK Stemming with spaCy------------------')
# Integrating NLTK Stemming with spaCy

# Load spaCy's small English model
ps = PorterStemmer()
doc = nlp("running runner easily run")
stems = [ps.stem(token.text) for token in doc]
print("spaCy Integrated Stemmed Tokens:", stems)

print('-----------------Lemmatization Strategies------------------')
# Lemmatization Strategies
# Lemmatization converts words into their base or dictionary forms by analyzing their context and morphology,
# resulting in more meaningful representations than simple stemming.
# NLTK uses the WordNetLemmatizer with the WordNet database,
# while spaCy provides built-in lemmatization integrated within its processing pipeline.

# NLTK Lemmatization Example
lemmatizer = WordNetLemmatizer()
text = "The striped bats are hanging on their feet for best"
words = nltk.word_tokenize(text)
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("NLTK Lemmatized Words:", lemmatized_words)

print('-----------------spaCy Lemmatization------------------')
text = "The striped bats are hanging on their feet for best"
doc = nlp(text)
lemmatized_tokens = [token.lemma_ for token in doc]
print("spaCy Lemmatized Tokens:", lemmatized_tokens)

print('-----------------Parsing and Chunking------------------')
# Parsing and Chunking
# Parsing involves analyzing a sentence's grammatical structure, 
# while chunking groups tokens into higher-level units like noun phrases. 
# NLTK employs rule-based parsers for chunking, 
# and spaCy utilizes statistical models to identify syntactic dependencies and extract phrase chunks.

# NLTK Parsing and Chunking Example
# Tokenize and tag parts of speech
text = "The quick brown fox jumps over the lazy dog."
tokens = nltk.word_tokenize(text)
tagged_tokens = nltk.pos_tag(tokens)

# Define a simple chunk grammar for noun phrases (NP)
grammar = "NP: {<DT>?<JJ>*<NN>}"

# Create a RegexpParser object and parse the tagged tokens
cp = nltk.RegexpParser(grammar)
parsed_tree = cp.parse(tagged_tokens)

print("NLTK Parsed Tree:")
print(parsed_tree)
# Uncomment the line below to visualize the tree (requires GUI support)
parsed_tree.pretty_print()

print('-----------------spaCy Noun Chunks------------------')
# Extract and print noun chunks using spaCy's built-in functionality
print("spaCy Noun Chunks:")
for chunk in doc.noun_chunks:
    print(chunk.text)

print('-----------------Hyponyms and Hypernyms Exploration------------------')
# Hyponyms and Hypernyms Exploration
# Hyponyms and hypernyms are semantic relationships that organize words hierarchically—hyponyms denote
# more specific terms (e.g., "poodle" for "dog"), 
# while hypernyms represent more general categories (e.g., "canine" for "dog").
# NLTK’s WordNet is a powerful resource for exploring these relationships. 
# Although spaCy doesn't natively extract hypernyms and hyponyms, 
# you can integrate its NLP capabilities with NLTK’s WordNet for extended lexical analysis.

# NLTK WordNet Example

# Choose a synset for the word 'dog'
dog_synset = wn.synset('dog.n.01')

# Retrieve hypernyms (more general terms)
hypernyms = dog_synset.hypernyms()
print("Hypernyms of 'dog':", hypernyms)

# Retrieve hyponyms (more specific terms)
hyponyms = dog_synset.hyponyms()
print("Hyponyms of 'dog':", hyponyms)

print('-----------------Integrating spaCy with NLTK for Lexical Exploration------------------')
# Integrating spaCy with NLTK for Lexical Exploration
# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")
doc = nlp("The dog barked at the mailman.")

# Find the token 'dog' and explore its lexical relationships using NLTK's WordNet
for token in doc:
    if token.text.lower() == 'dog':
        synsets = wn.synsets(token.text, pos=wn.NOUN)
        if synsets:
            synset = synsets[0]
            hypernyms = synset.hypernyms()
            hyponyms = synset.hyponyms()
            print(f"Token: {token.text}")
            print("Hypernyms:", hypernyms)
            print("Hyponyms:", hyponyms)

print('-----------------Named Entity Recognition (NER)------------------')
nlp = spacy.load("en_core_web_sm")
text = "Apple Inc. announced a new partnership with OpenAI at the annual Oscar Award event in California."

doc = nlp(text)

print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")


# ORG (Organization)
# GPE (Geopolitical Entity)
print('-----------------Gutenberg Text Preprocessing------------------')

# Select a text file from Gutenberg (e.g., 'shakespeare-hamlet.txt')
file_id = "shakespeare-hamlet.txt"
raw_text = gutenberg.raw(file_id)

# Step 1: Text Cleaning (Removing Gutenberg Header/Footer)
def clean_text(text):
    lines = text.split("\n") # break (enter - new line)
    start_idx, end_idx = 0, len(lines)

    # Removing Gutenberg boilerplate (First few and last few lines)
    for i, line in enumerate(lines):
        if "START OF THIS PROJECT GUTENBERG" in line:
            start_idx = i + 1
        if "END OF THIS PROJECT GUTENBERG" in line:
            end_idx = i
            break

    cleaned_lines = lines[start_idx:end_idx]
    cleaned_text = " ".join(cleaned_lines)
    return cleaned_text

text = clean_text(raw_text)

# Step 2: Lowercase
text = text.lower()

# Step 3: Tokenization
tokens = word_tokenize(text)

# Step 4: Remove Punctuation & Stopwords
stop_words = set(stopwords.words("english"))
tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

# Step 5: Stemming & Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(word) for word in tokens]
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

# Step 6: Convert back to text
stemmed_text = " ".join(stemmed_tokens)
lemmatized_text = " ".join(lemmatized_tokens)

# Output Results
print("Original Text (First 500 characters):\n", text[:500])
print("\nStemmed Text (First 500 characters):\n", stemmed_text[:500])
print("\nLemmatized Text (First 500 characters):\n", lemmatized_text[:500])