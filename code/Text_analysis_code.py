import re
import os
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.util import ngrams

# 1. Load a large English model from SpaCy
def load_language_model():
    """
    Load the SpaCy language model.
    Args:
        model_name (str): Name of the SpaCy model to load.
    Returns:
        spacy.Language: Loaded SpaCy language model.
    """
    return spacy.load("en_core_web_lg")


# 2. Read file content
def read_file(file_path):
    """
    Read the content of a file.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: File content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        
        # Adjust regex to better handle special cases and preserve periods, commas, apostrophes, hyphens
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\'-]', '', text)
        return text
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# 3. Process Text
def process_text(nlp, text, min_n, max_n):
    """
    Process text to extract n-grams, noun chunks, and named entities.
    Args:
        nlp (spacy.Language): SpaCy language model.
        text (str): Text to process.
        min_n (int): Minimum n-gram size.
        max_n (int): Maximum n-gram size.
    Returns:
        tuple: n-gram counts, noun chunks, and named entities.
    """
    doc         = nlp(text)
    ngram_freq  = {}
    noun_chunks = set(chunk.text.lower() for chunk in doc.noun_chunks)
    entities    = set(ent.text.lower() for ent in doc.ents)
    tokens      = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    for n in range(min_n, max_n + 1):
        for ngram_tuple in ngrams(tokens, n):
            if all(word.isalpha() for word in ngram_tuple):
                ngram_str = ' '.join(ngram_tuple).lower()
                ngram_freq[ngram_str] = ngram_freq.get(ngram_str, 0) + 1

    return ngram_freq, noun_chunks, entities


# 4. Analyze Common N-Grams
def analyze_common_ngrams(doc1_data, doc2_data, eng_dict, nlp):
    """
    Analyze common n-grams between two documents and calculate a weighted score for each.
    Args:
        doc1_data (tuple): N-gram, noun chunk, and named entity data for the first document.
        doc2_data (tuple): N-gram, noun chunk, and named entity data for the second document.
        eng_dict (dict): Dictionary of English word frequencies.
        nlp (spacy.Language): SpaCy language model for NLP processing.
    Returns:
        dict: Weighted n-grams that appear in both documents.
    """
    # Unpack n-gram, noun chunk, and named entity data
    doc1_ngrams, doc1_noun_chunks, doc1_entities = doc1_data
    doc2_ngrams, doc2_noun_chunks, doc2_entities = doc2_data
    res_ngrams = {}

    normalizing_factor = 0.01

    for ngram, f1 in doc1_ngrams.items():
        f2 = doc2_ngrams.get(ngram, 0)

        if f2 > 0:
            words = ngram.split(' ')
            freqs = [eng_dict.get(word.lower(), 0) for word in words]
            fe = sum(freqs) / len(words)

            doc = nlp(ngram)

            # Check for entity and noun chunk presence
            entity_presence = any(ent.text.lower() == ngram for ent in doc.ents)
            noun_chunk_presence = any(chunk.text.lower() == ngram for chunk in doc.noun_chunks)

            # POS tagging to exclude verbs and non-noun/adjective words
            pos_tags = [token.pos_ for token in doc]
            contains_verb = 'VERB' in pos_tags
            contains_non_noun_adjective = any(tag not in {'NOUN', 'PROPN', 'ADJ'} for tag in pos_tags)

            if not contains_verb and not contains_non_noun_adjective:
                difference_f1_fe = (f1 - fe) ** 2 * normalizing_factor
                difference_f2_fe = (f2 - fe) ** 2 * normalizing_factor

                base_multiplier = 1 + 0.5 * (int(noun_chunk_presence) + int(entity_presence))
                weight = (difference_f1_fe + difference_f2_fe) * base_multiplier

                res_ngrams[ngram] = {
                    'f1': f1,
                    'f2': f2,
                    'fe': fe,
                    'noun': noun_chunk_presence,
                    'ner': entity_presence,
                    'weight': weight
                }

    return res_ngrams


# 5. Normalize Frequencies
def normalize_frequencies(freq_dict):
    """
    Normalize the weights in the n-gram dictionary by applying the square root to each weight.
    Args:
        freq_dict (dict): N-gram dictionary with weights.
    Returns:
        dict: N-gram dictionary with normalized weights.
    """
    return {word: {**info, 'weight': np.sqrt(info['weight'])} for word, info in freq_dict.items()}


# 6. Print N-Gram Frequencies
def print_ngram_frequencies(ngram_freq, n, top_n=100):
    """
    Print the frequencies of n-grams.
    Args:
        ngrams (dict): N-gram dictionary with weights.
        n (int): N-gram size.
    """    
    sorted_ngrams = sorted(ngram_freq.items(), key=lambda x: (-x[1]['weight'], x[0]))[:top_n]
    print(f"Top {top_n} {n}-grams by weight:")
    for ngram, data in sorted_ngrams:
        print(f"{ngram}: f1={data['f1']}, f2={data['f2']}, fe={data['fe']}, weight={data['weight']}, noun={data['noun']}, ner={data['ner']}")
    print()  

    
# 7. Create Word Cloud
def create_wordcloud(freq_dict, title, output_directory, filename, top_n=100):
    """
    Create a word cloud from the n-gram weights.
    Args:
        ngrams (dict): N-gram dictionary with weights.
        title (str): Title for the word cloud plot.
        output_directory (str): Directory to save the plot.
        filename (str): File name for the plot.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    sorted_freq_dict = dict(sorted(freq_dict.items(), key=lambda x: -x[1]['weight'])[:top_n])
    
    if not sorted_freq_dict:
        print("No words to plot. The frequency dictionary is empty.")
        return
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies({k: v['weight'] for k, v in sorted_freq_dict.items()})
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    
    output_path = os.path.join(output_directory, filename + '.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Word cloud saved to {output_path}")


    
if __name__ == '__main__':
    nlp_model = load_language_model()
    
    # Paths for the documents
    doc1_path = ""
    doc2_path = ""
    output_directory = ""
    
    # Set the range of n-grams here
    min_n = 1
    max_n = 3

    doc1_text = read_file(doc1_path)
    doc2_text = read_file(doc2_path)

    if doc1_text and doc2_text:
        doc1_data = process_text(nlp_model, doc1_text, min_n, max_n)
        doc2_data = process_text(nlp_model, doc2_text, min_n, max_n)

        # Eng Freq Path
        df = pd.read_csv("english_ngram_freq.csv")
        
        eng_dict    = df.set_index('word')['count'].to_dict()
        res_ngrams  = analyze_common_ngrams(doc1_data, doc2_data, eng_dict, nlp_model)
        res_ngrams_normalized = normalize_frequencies(res_ngrams)
        
        for n in range(min_n, max_n + 1):
            ngram_specific = {k: v for k, v in res_ngrams_normalized.items() if len(k.split(' ')) == n}
            create_wordcloud(ngram_specific, f'Word Cloud for {n}-grams', output_directory, f'ngram_{n}_wordcloud')
            print_ngram_frequencies(ngram_specific, n)