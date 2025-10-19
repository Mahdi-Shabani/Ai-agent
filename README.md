## Job Description Analyzer
This project is a Python-based tool designed to extract and analyze required skills from job postings using Natural Language Processing (NLP) techniques. It leverages libraries like NLTK for preprocessing and SpaCy for Named Entity Recognition (NER) to identify trends and in-demand skills across various industries.
## Project Overview

Objective: Extract and categorize skills from job descriptions to identify trends and analyze the most demanded skills in industries like IT, Finance, and Marketing.
Tools Used: Python, NLTK, SpaCy, Pandas.
Output: A dataset with processed descriptions, extracted skills, and a ranking of top skills per industry.

## Features

Preprocesses job descriptions (tokenization, stopword removal, lowercase conversion).
Uses NER to extract skills and technical terms.
Analyzes skill frequency and categorizes them by industry.
Provides a comparison of top skills across selected industries.

## Prerequisites

Python 3.x
Required libraries:
pandas
nltk
spacy


Download SpaCy model: en_core_web_sm (run python -m spacy download en_core_web_sm)
Dataset: data.csv (available via the provided link below)

## Installation

Install dependencies:pip install -r requirements.txt

(Create a requirements.txt file with: pandas nltk spacy and add it to the repo.)
Download the dataset:wget https://raw.githubusercontent.com/binoydutt/Resume-Job-Description-Matching/refs/heads/master/data.csv



## Usage

Run the notebook or scripts in a Python environment (e.g., Google Colab or VS Code).
Follow the steps in the code to:
Load and preprocess the dataset.
Extract skills using NER.
Analyze and categorize skills by industry.


Check the output for top skills and industry comparisons.

## Example Output

Top 10 skills might include: "Python", "SQL", "Project Management".
Industry comparison: IT focuses on "Coding", Marketing on "Communication".


Dataset source: [https://github.com/binoydutt/Resume-Job-Description-Matching]
Inspired by NLP learning and job market analysis needs.
