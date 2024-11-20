# LanguagePatternAnalyzer

**LanguagePatternAnalyzer** is a comprehensive project that leverages Natural Language Processing (NLP) and statistical techniques to identify and analyze hidden language patterns across multiple documents. This project aims to uncover insightful relationships and trends in textual data, enabling a deeper understanding of language usage, semantics, and document comparisons.

## Overview

This repository contains a Python tool for comparative text analysis, focusing on extracting and comparing n-grams, noun chunks, and named entities from text documents. The tool leverages natural language processing (NLP) techniques and statistical analysis to provide insights into language patterns and relationships between documents.

## Features

- **N-Gram Analysis:** Implement various n-gram models to capture and analyze word combinations within documents.
- **Statistical Methods:** Utilize statistical techniques to quantify language patterns and evaluate their significance across different texts.
- **Data Visualization:** Generate visual representations of language patterns, making it easier to identify trends and anomalies.
- **Cross-Document Comparison:** Analyze multiple documents simultaneously to uncover hidden similarities and differences in language use.
- **Explainability:** Provide clear insights into the identified patterns, making it accessible for both technical and non-technical audiences.

## Folder Structure

- **Code:** Contains Python scripts for data scraping, text analysis, and other utilities.
  - `scrape.py`: Python script for scraping text data from Wikipedia.
  - `text_analysis_code.py`: Main Python script for text analysis.
  - > **Note:** The English word frequency data (`eng_freq.csv`) is not included in this repository due to its size (150 MB). You can find this data online from sources like COCA or similar databases; it's a simple Excel file of English word frequencies.

- **Data:** Contains text data categorized by topics (e.g., presidents, scientists).
  - **presidents:** Directory containing text files for each president.
  - **scientists:** Directory containing text files for each scientist.
  - **Textanalysis_res:** Subfolder containing results of text analysis comparisons.

- **Documentation:** Contains documentation and additional resources.
  - `Text_Analysis_Documentation.pdf`: PDF document with detailed documentation.

- **README.md:** Main README file providing an overview of the repository and instructions for usage.

## Technologies Used

- **Programming Languages:** Python
- **Libraries:** NLTK, SpaCy, Matplotlib, Pandas, NumPy, etc.
- **Data Sources:** Custom datasets collected from various text sources (e.g., articles, reports, social media).

## Usage

### Setup

- Install required libraries using the following commands:
  ```bash
  pip install spacy pandas numpy wordcloud matplotlib nltk
  python -m spacy download en_core_web_lg
- Ensure the necessary data files are available in the Data directory.

### Running the Tool

- Open and run the text_analysis_code.py script to perform text analysis.
- Adjust parameters such as min_n and max_n for the desired range of n-grams.
- View the results in the Textanalysis_res folder, including word cloud images and top n-grams lists.
