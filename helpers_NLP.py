'''
Helpers for text classification project on identifying text posts on social media/forums with signs
of psychological distress.
'''
# Essentials
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Optional, List
import datetime
import time

# API
import requests
import praw

# Text related processing
import re
import string
import nltk
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# sklearn modelling and pipelines
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin                 # For custom transformers, keeping consistent interface to be used with sklearn pipeline and gridsearchCV
from sklearn.preprocessing import StandardScaler

# sklearn models
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# XGBoost
from xgboost import XGBClassifier

# HuggingFace Models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

# Pytorch for BERT
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import onnxruntime as ort

# For progress bars during NN training
from tqdm.auto import tqdm

from scipy.sparse import csr_matrix


def get_reddit_posts2(sub_reddit: str, post_type: Optional[str]='new',
                     client_id: Optional[str]=None,
                     client_secret: Optional[str]=None,
                     user_agent: Optional[str]=None,
                     nsfw: Optional[bool] = True,
                     limit: Optional[int] = None
                     ):
    '''
    To scrape subreddit for up to 1000 posts, including comments and subcomments (retrieved recursively)
    sub_reddit: Name of subreddit. Follows /r/ in url
    post_type: Default 'new'. Can be 'new', 'hot', 'top', or 'rising'
    nsfw: False to exclude NSFW material. Default True. 
    limit: Number of posts to get from subreddit. Default None for max of up to 1000 posts.
    '''

    # Authenticate with Reddit
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent  # Describe your app
    )

    # Choose subreddit to scrape
    subreddit = reddit.subreddit(sub_reddit)

    # Retrieve all posts from the subreddit.
    if post_type == 'new':
        all_posts = subreddit.new(limit=limit)
    elif post_type == 'hot':
        all_posts = subreddit.hot(limit=limit)
    elif post_type == 'top':
        all_posts = subreddit.top(limit=limit)
    elif post_type == 'rising':
        all_posts = subreddit.rising(limit=limit)
    else:
        raise ValueError(f"Unrecognized post_type: {post_type}", nsfw=nsfw)

    return list(all_posts)

def get_post_texts2(all_posts: list,
                   get_all_comments: Optional[bool]=False,
                   score_threshold: Optional[int] = -np.inf):
    '''
    Using raw subreddit material from praw, provide list of lists of bundled post text and comments
    all_posts: All posts obtained by praw
    score_threshold: Threshold score to decide whether to include post/comment/replies in dataset. Default negative infinity to include all
    get_all_comments: True to use replace_more to get all comments. Takes significantly longer, setting 1s delay here for each morecomments replacement to prevent getting banned
    '''

    # Get bundled post body and associated comments in a list
    all_post_texts = []
    counter=-1

    # Extract post title, body, comments and subcomments for each post
    for post in all_posts:
        counter+=1
        if counter%50==0:
            print(f'{counter} of {len(all_posts)} posts processed.')
        post_text = []
        if post.score < score_threshold:
            continue
        post_text.append(post.title)                        # Add title of post
        post_text.append(post.selftext)                     # Add main body of post
        
        if get_all_comments:
            post.comment_sort = "top"
            post.comments.replace_more(limit=None)          # Replace morecomments objects with more comments
            time.sleep(1)
        else:
            post.comments.replace_more(limit=0)             # Remove morecomments object, which will otherwise lead to error later.

        post_agg_comments = process_comments2(post.comments,score_threshold)     # Get all comments and subcomments
        post_text += post_agg_comments
        all_post_texts.append(post_text)

    return all_posts,all_post_texts

def process_comments2(comments, score_threshold):
    '''
    Recursively get comments and replies
    comments: Comment object from PRAW
    score_threshold: Threshold score to decide whether to include post/comment/replies in dataset. Default negative infinity to include all
    '''
    # Function to allow comments to be recursively drawn from replies (sub-comments) to comments
    def process_replies(comment, replies):
        if comment.score < score_threshold:
            return replies
        if len(comment.replies) > 0:
            for reply in comment.replies:
                replies.append(reply.body)
                process_replies(reply, replies)
        return replies
    
    agg_comments = []                                                                # Bundle all texts in a list
    for comment in comments:
        if comment.score >= score_threshold:
            comment_and_replies = [comment.body]
            replies = process_replies(comment, [])
            if replies:
                comment_and_replies.extend(replies)
            agg_comments.append(" ".join(comment_and_replies))                     # Combine comment and its replies with delimiter
    return agg_comments

def make_df_with_url(all_posts,all_posts_texts,category_name='new'):
    '''
    Create dataframe with from extracted text posts.
    all_posts: List of posts.
    all_posts_texts: List of texts from each post, including title, body, comments and all subcomments/labels
    '''
    id = [post.id for post in all_posts]
    title = [post.title for post in all_posts]
    body = [post.selftext for post in all_posts]
    score = [post.score for post in all_posts]
    num_comments = [post.num_comments for post in all_posts]
    category = [category_name for _ in all_posts]
    comments = [' '.join(comment[2:]) for comment in all_posts_texts]
    urls = [post.url for post in all_posts]
    df = pd.DataFrame({'id':id,'title':title,'body':body,'score':score,'num_comments':num_comments,'category':category,'comments':comments,'url':urls})

    df.drop_duplicates(subset=['id'],keep='first',inplace=True)
    df.reset_index(drop=True,inplace=True)

    df['comments'] = df['comments'].str.lstrip('[').str.rstrip(']').str.lstrip('\'').str.rstrip('\'').str.lstrip('\"').str.rstrip('\"')
    df['comments'] = df['comments'].replace('', np.nan)
    df[['title','body','comments','url']] = df[['title','body','comments','url']].fillna('')
    return df


def preprocess_regular(df_raw,min_num_words=None,split_length=100,remove_punc=True):
    '''
    Preprocessing performed for traditional machine learning approaches requiring matrices of a specified format.
    This function provides the option to remove punctuation, remove stop words, perform stemming, etc.
    df_raw: Dataframe of specified format, with text (string) columns including 'title', 'body' and 'comments', and a 'target' column of either 'mentalhealth' or 'casualconversation'
    min_num_words: Minimum number of words in post. Those below this limit are dropped. None for no limit.
    split_length: Split long text sequences into multiple data points of approximately this length. None for no splitting.
    remove_punch: Set to True to remove punctuation.
    '''
    # Prepping functions for later pre-processing
    def clean_text(text,remove_punc=True):
        text = "".join([word.lower() for word in text if word not in string.punctuation]) # Remove all punctuation
        tokens = re.split('\W+', text)                                                    # Split by whitespace
        text = ' '.join([ps.stem(word) for word in tokens if word not in stopwords])      # Stem, remove stop words
        return text
    
    # Function to split long sequences into multiple shorter sequences 
    def split_long_text(row,length_limit=100, split_length=100):
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
        
    df = df_raw.copy()
    df.drop(columns=['score','num_comments','category','url'],inplace=True)

    # Lump all texts together
    df['alltext'] = df['title'] + ' ' + df['body'] + ' ' + df['comments']
    df.drop(columns=['title','body','comments'],inplace=True)
    df['alltext'] = df['alltext'].str.replace(r'\s+', ' ', regex=True)                    # Remove all non-space whitespace characters

    # Encode target
    if 'target' in df.columns:
        df['target'] = df['target'].map(lambda x: 1 if x=='mentalhealth' else 0)
        
    # Get word count
    df['word_count'] = df['alltext'].map(lambda x: len(x.split()))           
    # Drop short sequences
    if min_num_words is not None:
        df = df.loc[df['word_count']>min_num_words,:]
            
    # Split long texts into multiple short texts
    if split_length is not None:
        df['alltext_split'] = df.apply(split_long_text, axis=1,length_limit=100,split_length=split_length)
        df = df.explode('alltext_split')

        if min_num_words is not None:
            df['word_count'] = df['alltext_split'].map(lambda x: len(x.split()))           # Update word count
            df = df.loc[df['word_count']>min_num_words,:]                                  # Drop all short sequences below 20 words long
        df.drop(columns=['alltext','word_count'],inplace=True)                         # Drop unwanted columns
        df.rename(columns={'alltext_split':'alltext'},inplace=True)
        df.reset_index(drop=True,inplace=True)

    # Get English stop words
    stopwords = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()
    df['alltext'] = df['alltext'].map(lambda x: clean_text(x,remove_punc=remove_punc))
    if 'target' in df.columns:
        display(df['target'].value_counts())                                           # Check for possible class imbalance after splitting. Looks ok.
    return df

def combine_preds_regular(pred_proba,id_test,y_test=None,return_probs=False):
    '''
    Compare predicted values against test values for traditional machine learning models. This function can
    account for inference for long text sequences that have been split into multiple child sequences
    and inferred on separately. In this case, the probabilities for each child sequence are averaged to
    provide the final prediction.
    pred_proba: Predicted probability for each text sequence, possibly having been split up.
    y_test: Original sequence of observed values
    id_test: id of text posts to identify child text sequences of each post, if splitting had been done.
    return_probs: True to return mean probability instead of prediction based on 0.5 threshold.
    '''
    y_pred_grouped = pd.DataFrame(pred_proba,columns=['casconv','menhealth'])
    y_pred_grouped['id'] = id_test.reset_index(drop=True)
    y_pred_grouped = y_pred_grouped.groupby('id').mean()
    if not return_probs:
        y_pred_grouped = y_pred_grouped['menhealth']>0.5
        y_pred_grouped = y_pred_grouped.astype('int64')

    y_pred_grouped.sort_index(inplace=True)

    if y_test is not None:
        y_test_grouped = y_test.copy()
        y_test_grouped = pd.DataFrame(y_test).reset_index(drop=True)
        y_test_grouped['id'] = id_test.reset_index(drop=True)
        y_test_grouped.drop_duplicates(subset='id',inplace=True)
        y_test_grouped.set_index('id',inplace=True)
        y_test_grouped.sort_index(inplace=True)
        return y_test_grouped, y_pred_grouped
    else:
        return y_pred_grouped

def combine_preds_BERT(pred_proba,id_test,y_test=None,return_probs=False):
    '''
    Compare predicted values against test values for BERT model. This function can
    account for inference for long text sequences that have been split into multiple child sequences
    and inferred on separately. In this case, the probabilities for each child sequence are averaged to
    provide the final prediction.
    pred_proba: Predicted probability for each text sequence, possibly having been split up.
    y_test: Original observed values
    id_test: id of text posts to identify child text sequences of each post, if splitting had been done.
    return_probs: True to return mean probability instead of prediction based on 0.5 threshold.
    '''
    try:
        y_pred_grouped = pd.DataFrame(torch.cat(pred_proba).cpu().numpy(),columns=['casconv','menhealth'])
    except:
        y_pred_grouped = pd.DataFrame(pred_proba,columns=['casconv','menhealth'])
    y_pred_grouped['id'] = id_test.reset_index(drop=True)
    y_pred_grouped = y_pred_grouped.groupby('id').mean()
    if not return_probs:
        y_pred_grouped = y_pred_grouped['menhealth']>0.5
        y_pred_grouped = y_pred_grouped.astype('int64')

    y_pred_grouped.sort_index(inplace=True)

    if y_test is not None:
        y_test_grouped = y_test.copy()
        y_test_grouped = pd.DataFrame(y_test).reset_index(drop=True)
        y_test_grouped['id'] = id_test.reset_index(drop=True)
        y_test_grouped.drop_duplicates(subset='id',inplace=True)
        y_test_grouped.set_index('id',inplace=True)
        y_test_grouped.sort_index(inplace=True)

        # Filter y_test based on samples in y_pred
        try:
            y_test_grouped = pd.concat([y_pred_grouped.to_frame(),y_test_grouped],axis=1).dropna()['target']
        except:
            y_test_grouped = pd.concat([y_pred_grouped,y_test_grouped],axis=1).dropna()['target']
        return y_test_grouped, y_pred_grouped
    else:
        return y_pred_grouped

def create_confusion_matrix_and_class_report(y_test,y_pred,make_confusion=False,output_dict=False):
    '''
    Create confusion matrix and classification report given predicted and observed values.
    y_test: Original observed values
    y_pred: Predicted values
    make_confusion: Set True to show confusion matrix
    output_dict: Set True to return metric values from classification report, instead of just printing it.
    '''
    if make_confusion:
        # Generate confusion matrix
        cm = confusion_matrix(y_test.to_numpy(), y_pred.to_numpy())

        # Plot confusion matrix
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False,
                    xticklabels=['Casual Conversation','Mental Health'], yticklabels=['Casual Conversation','Mental Health'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

    # Generate classification report
    report = classification_report(y_test.to_numpy(), y_pred.to_numpy(),
                                target_names=['Casual Conversation','Mental Health'])

    if output_dict:
        report = classification_report(y_test.to_numpy(), y_pred.to_numpy(),
                                target_names=['Casual Conversation','Mental Health'],output_dict=output_dict)
        return report
    else:
        print("Classification Report:\n", report)

def preprocess_BERT(df_raw,min_num_words=None,split_length=100):
    '''
    Preprocessing performed for BERT model. Splitting
    df_raw: Dataframe of specified format, with text (string) columns including 'title', 'body' and 'comments', and a 'target' column of either 'mentalhealth' or 'casualconversation'
    min_num_words: Minimum number of words in post. Those below this limit are dropped. None for no limit.
    split_length: Split long text sequences into multiple data points of approximately this length. None for no splitting.
    '''
    # Function to split long sequences into multiple shorter sequences 
    def split_long_text(row,length_limit=100, split_length=100):
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
        
    df = df_raw.copy()
    df.drop(columns=['score','num_comments','category','url'],inplace=True)

    # Lump all texts together
    df['alltext'] = df['title'] + ' ' + df['body'] + ' ' + df['comments']
    df.drop(columns=['title','body','comments'],inplace=True)
    df['alltext'] = df['alltext'].str.replace(r'\s+', ' ', regex=True)                  # Remove all non-space whitespace characters

    if 'target' in df.columns:
        # Encode target
        df['target'] = df['target'].map(lambda x: 1 if x=='mentalhealth' else 0)
        
    # Get word count
    df['word_count'] = df['alltext'].map(lambda x: len(x.split()))           
    # Drop short sequences
    if min_num_words is not None:
        df = df.loc[df['word_count']>min_num_words,:]

    # Split long texts into multiple short texts
    if split_length is not None:
        df['alltext_split'] = df.apply(split_long_text, axis=1,length_limit=split_length,split_length=split_length)
        df = df.explode('alltext_split')

        df['word_count'] = df['alltext_split'].map(lambda x: len(x.split()))           # Update word count
        if min_num_words is not None:
            df = df.loc[df['word_count']>min_num_words,:]                                             # Drop all sequences below 20 words long
        df.drop(columns=['alltext','word_count'],inplace=True)                         # Drop unwanted columns
        df.rename(columns={'alltext_split':'alltext'},inplace=True)
        df.reset_index(drop=True,inplace=True)

    if 'target' in df.columns:
        display(df['target'].value_counts())                                           # Check for possible class imbalance after splitting
    
    return df

# Create dataloader
def create_dataset_dict(tokenizer, X_test, X_train=None):
    '''
    Simple function to create a DatasetDict object, either holding both train and test datasets, or just the test dataset.
    tokenizer: a Tokenizer object from the pre-trained google-bert transformer
    X_test: A dataframe consisting of test features and labels
    X_train: A dataframe consisting of training features and labels
    '''
    def create_dataset(X):
        dataset = tokenizer(list(X.loc[:,'alltext']),padding='max_length', truncation=True, return_tensors='pt',max_length=387)

        # Create a Dataset object
        if 'target' in X.columns:
            dataset = Dataset.from_dict({
                'input_ids': torch.tensor(dataset['input_ids']),
                'attention_mask': torch.tensor(dataset['attention_mask']),
                'token_type_ids': torch.tensor(dataset['token_type_ids']),
                'labels': torch.tensor(X['target'].values)
                })
        else:
            dataset = Dataset.from_dict({
                'input_ids': torch.tensor(dataset['input_ids']),
                'attention_mask': torch.tensor(dataset['attention_mask']),
                'token_type_ids': torch.tensor(dataset['token_type_ids'])
                })
        return dataset
    
    test_dataset = create_dataset(X_test)
    if X_train is not None:
        train_dataset = create_dataset(X_train)
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
            })
    else:
        dataset_dict = DatasetDict({
            'test': test_dataset
            })
    dataset_dict.set_format("torch")
    return dataset_dict

# Function to convert a PyTorch tensor to a NumPy array
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Function to perform inference on the whole evaluation dataloader
def run_inference(pred_dataloader,ort_session,num_batches=None):
    all_preds_logits = []
    
    # Iterate over the dataloader
    for i,batch in enumerate(pred_dataloader):
        input_ids_batch = batch['input_ids']
        attention_mask_batch = batch['attention_mask']

        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(input_ids_batch),
            ort_session.get_inputs()[1].name: to_numpy(attention_mask_batch)
        }
        
        preds_logits = ort_session.run(None, ort_inputs)
        all_preds_logits.append(preds_logits[0])  # Assuming preds_logits is a list, and we're interested in the first item

        if (num_batches is not None):
            if i == num_batches:
                break

    # Concatenate all predictions
    all_preds_logits = np.concatenate(all_preds_logits, axis=0)
    all_preds_proba = 1 / (1 + np.exp(-all_preds_logits))
    return all_preds_proba

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for Word2Vec vectorizer, suitable for use with sklearn pipeline
    '''
    def __init__(self, size=100, window=5, min_count=1):
        self.size = size
        self.window = window
        self.min_count = min_count
    
    def fit(self, X, y=None):
        sentences = [text.split() for text in X]
        self.model = Word2Vec(sentences, vector_size=self.size, window=self.window, min_count=self.min_count)
        return self
    
    def transform(self, X):
        return np.array([np.mean([self.model.wv[word] for word in text.split() if word in self.model.wv] or [np.zeros(self.size)], axis=0) for text in X])


class FastTextVectorizer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for FastText vectorizer, suitable for use with sklearn pipeline
    '''
    def __init__(self, size=100, window=5, min_count=1, workers=4):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
    
    def fit(self, X, y=None):
        sentences = [text.split() for text in X]
        self.model = FastText(sentences, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers)
        return self
    
    def transform(self, X):
        return np.array([np.mean([self.model.wv[word] for word in text.split() if word in self.model.wv] or [np.zeros(self.size)], axis=0) for text in X])


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for Doc2Vec vectorizer, suitable for use with sklearn pipeline
    '''
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
    
    def fit(self, X, y=None):
        tagged_data = [TaggedDocument(words=text.split(), tags=[i]) for i, text in enumerate(X)]
        self.model = Doc2Vec(tagged_data, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers, epochs=self.epochs)
        return self
    
    def transform(self, X):
        return np.array([self.model.infer_vector(text.split()) for text in X])
    
# Custom transformer to sort indices
class SortIndicesAndConvertToTensor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, csr_matrix):
            raise ValueError("Expected input to be a csr_matrix")
        X.sort_indices()
        return X
