import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from helpers_st import *

# Prepping functions for later pre-processing
def clean_text(text,remove_punc=True):
    stopwords = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()
    text = "".join([word.lower() for word in text if word not in string.punctuation]) # Remove all punctuation
    tokens = re.split('\W+', text)                                                    # Split by whitespace
    text = ' '.join([ps.stem(word) for word in tokens if word not in stopwords])      # Stem, remove stop words
    return text

# Function to split long sequences into multiple shorter sequences 
def split_long_text(row,length_limit=500, split_length=500):
    text = row['alltext']
    if len(text.split()) <= length_limit:
        return text
    else:
        words = text.split()
        num_splits = (len(words) - 1) // split_length + 1  # Calculate number of split_length sequences
        rows = []
        for i in range(num_splits-1):
            start_idx = i * split_length
            end_idx = (i + 1) * split_length
            new_text = ' '.join(words[start_idx:end_idx])
            rows.append(new_text)
        start_idx = (num_splits-1) * split_length
        new_text = ' '.join(words[start_idx:])    
        rows.append(new_text)
        return rows
        
min_num_words = 20

# Preload models
# Load Naive Bayes Model
with open(r'../models/MultinomialNB_TfidfVectorizer.pkl','rb') as f:
    model_NB = pickle.load(f)
st.session_state.model_NB = model_NB

# Load BERT
st.session_state.ort_session = ort.InferenceSession(r"../models/BERT_reddit_quantized.onnx")


st.title('Mental Health Monitoring - Single Text Input')
st.write('Enter the text you want to classify below.')

# Text input
user_input = st.text_area('Enter text here')

# Model selection
model_options = ['Naive Bayes', 'BERT']
model_type = st.selectbox('Select model for classification:', model_options)

if st.button('Classify'):
    if user_input:
        if model_type == 'Naive Bayes':
            df_input = pd.DataFrame({'alltext': [clean_text(user_input,remove_punc=True)]})
            preds_proba = st.session_state.model_NB.predict_proba(df_input['alltext'])

            st.write('Prediction Probability:')
            st.write(f'Mental Health: {preds_proba[0][1]:.2f}')
            st.write(f'No Mental Health: {preds_proba[0][0]:.2f}')

        elif model_type == 'BERT':
            df_input = pd.DataFrame({'alltext': [user_input]})
            # Split long texts into multiple short texts
            df_input['alltext_split'] = df_input.apply(split_long_text, axis=1,length_limit=500,split_length=500)
            df = df_input.explode('alltext_split')

            df['word_count'] = df['alltext_split'].map(lambda x: len(x.split()))           # Update word count
            if min_num_words is not None:
                df = df.loc[df['word_count']>min_num_words,:]                                             # Drop all sequences below 20 words long
            df.drop(columns=['alltext','word_count'],inplace=True)                         # Drop unwanted columns
            df.rename(columns={'alltext_split':'alltext'},inplace=True)
            df.reset_index(drop=True,inplace=True)

            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
            dataset_dict = create_dataset_dict(tokenizer, df)
            pred_dataloader = DataLoader(dataset_dict['test'], batch_size=1)

            preds_proba = run_inference(pred_dataloader, st.session_state.ort_session, num_batches=None)
            preds_proba = preds_proba.mean(axis=0)

            st.write('Prediction Probability:')
            st.write(f'With mental health distress signals: {preds_proba[1]:.2f}')
            st.write(f'No mental health distress signals: {preds_proba[0]:.2f}')