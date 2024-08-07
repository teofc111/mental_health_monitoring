import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from helpers_st import *



# Set title and intro text
st.title('Mental Health Monitoring')
st.write('Welcome to the social media mental health monitoring platform. Please retrieve posts required for batch classification. Alternatively, select the \'Text\' pane on the left to experiment with single text classification')

st.subheader('Load Data')
st.write('Please use one of the following two options to load the required posts.')
# Create two columns with a divider
col1, col2, col3 = st.columns([3,1,3])
df = None # Initialize


# Preload models
# Load Naive Bayes Model
with open(r'../models/MultinomialNB_TfidfVectorizer.pkl','rb') as f:
    model_NB = pickle.load(f)
st.session_state.model_NB = model_NB

# Load BERT
st.session_state.ort_session = ort.InferenceSession(r"../models/BERT_reddit_quantized.onnx")

with col1:
    # Option to upload a CSV file
    st.write('Option 1: Upload a csv file containing posts of interest')
    uploaded_file = st.file_uploader("Upload csv.", type="csv")
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        df.drop_duplicates(subset=['id'],keep='first',inplace=True)
        df.reset_index(drop=True,inplace=True)
        df[['title','body','comments','url']] = df[['title','body','comments','url']].fillna('')

        # Processing to strip texts 
        df['comments'] = df['comments'].str.lstrip('[').str.rstrip(']').str.lstrip('\'').str.rstrip('\'').str.lstrip('\"').str.rstrip('\"')
        df['comments'] = df['comments'].replace('', np.nan)
        df[['title','body','comments']] = df[['title','body','comments']].fillna('')
        df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
        if 'df_sample' not in st.session_state:
            df_sample = df.sample(min(df.shape[0], 20))  # Limit to 20 posts maximum
            st.session_state.df = df
            st.session_state.df_sample = df_sample
        
with col3:
    # Option to run custom code to retrieve a CSV file
    st.write('Option 2: Retrieve the latest posts')
    # Get API details from user
    CLIENT_ID = st.text_input("Client ID", value=None, type="password")
    CLIENT_SECRET = st.text_input("Client Secret", value=None, type="password")
    USER_AGENT = st.text_input("User Agent",value=None)

    # Get post details from users
    sub_reddit = st.text_input("Sub-reddit", value='mentalhealth')
    num_posts = st.number_input("Number of posts to retrieve", min_value=1, value=10)

    if st.button('Retrieve',use_container_width=True):
        with st.spinner('Processing...'):
            if CLIENT_ID is None: # If not supplied, use those stored in env
                from dotenv import load_dotenv
                # Define keys
                load_dotenv('../.env')
                CLIENT_ID = os.getenv("CLIENT_ID")
                CLIENT_SECRET = os.getenv("CLIENT_SECRET")
                USER_AGENT = os.getenv("USER_AGENT")

            all_posts_hot = get_reddit_posts2(sub_reddit = sub_reddit, client_id = CLIENT_ID, client_secret = CLIENT_SECRET,
                             user_agent = USER_AGENT, post_type = 'hot',nsfw = True,limit=num_posts)
            all_posts_hot,all_post_texts_hot = get_post_texts2(all_posts_hot,get_all_comments=False)
            df = make_df_with_url(all_posts_hot,all_post_texts_hot)

            df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
            if 'df_sample' not in st.session_state:
                df_sample = df.sample(min(df.shape[0], 20))  # Limit to 20 posts maximum
                st.session_state.df = df
                st.session_state.df_sample = df_sample

            st.success('Retrieval successful')

# Filename input and download button
if 'df' in st.session_state:
    st.write('---')
    st.dataframe(st.session_state.df)
    # Provide a default value 'dataframe' for the filename input
    file_name = st.text_input('Enter the filename (without .csv)', 'dataframe')
    if st.download_button(
        label="Download data as CSV",
        data=convert_df_to_csv(st.session_state.df),
        file_name=f"{file_name}.csv",
        mime='text/csv',
    ):
        st.success('DataFrame downloaded.')

if 'df_sample' in st.session_state:
    st.subheader('Classify texts')
    st.write('Select model for classification')
    model_type = None
    model_options = ['Naive Bayes', 'BERT']
    model_type = st.selectbox('Select an option:', model_options)

    # Button to resample
    if st.button('Resample'):
        st.session_state.df_sample = st.session_state.df.sample(min(st.session_state.df.shape[0], 20))
        if 'preds_proba_NB' in st.session_state:
            del st.session_state.preds_proba_NB
        if 'preds_proba_BERT' in st.session_state:
            del st.session_state.preds_proba_BERT


    if model_type is not None:
        if model_type == 'Naive Bayes':
            X_test = preprocess_regular(st.session_state.df_sample,min_num_words=None,split_length=None,remove_punc=True)
            if 'preds_proba_NB' not in st.session_state:
                st.session_state.preds_proba_NB = st.session_state.model_NB.predict_proba(X_test['alltext'])
            
            # Slider with default values
            st.write('Adjust the threshold for classification. A higher threshold means the model will only flag posts as positive (with mental health markers) when it is very confident, resulting in fewer positive classifications. Conversely, a lower threshold will lead to more instances being classified as positive, even if the model is less certain.')
            threshold = None
            threshold = st.slider('Select threshold', min_value=0, max_value=100, value=50, step=1)

            if threshold is not None:
                preds = st.session_state.preds_proba_NB >threshold/100
                st.write(f'Posts with mental health markers (total: {np.sum(preds[:,1])}):')
                print('\n\n\n')
                print(X_test)
                st.write(st.session_state.df_sample.loc[preds[:,1],['body','url']])
                st.write(f'Posts without mental health markers (total: {np.sum(~preds[:,1])}):')
                st.write(st.session_state.df_sample.loc[~preds[:,1],['body','url']])

        elif model_type == 'BERT':
            X_test = preprocess_BERT(st.session_state.df_sample,min_num_words=20,split_length=500)
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
            dataset_dict = create_dataset_dict(tokenizer, X_test)
            pred_dataloader = DataLoader(dataset_dict['test'], batch_size=8)

            if 'preds_proba_BERT' not in st.session_state:
                preds_proba = run_inference(pred_dataloader, st.session_state.ort_session, num_batches=None)
                st.session_state.preds_proba_BERT = combine_preds_BERT(preds_proba, X_test['id'].iloc[:preds_proba.shape[0]], return_probs=True)

            # Slider with default values
            st.write('Adjust the threshold for classification. A higher threshold means the model will only flag posts as positive (with mental health markers) when it is very confident, resulting in fewer positive classifications. Conversely, a lower threshold will lead to more instances being classified as positive, even if the model is less certain.')
            threshold = None
            threshold = st.slider('Select threshold', min_value=0, max_value=100, value=50, step=1)

            if threshold is not None:
                preds = st.session_state.preds_proba_BERT >threshold/100
                preds = preds.to_numpy()
                st.write(f'Posts with mental health markers (total: {np.sum(preds[:,1])}):')
                st.write(st.session_state.df_sample.iloc[:preds.shape[0],:].loc[preds[:,1],['body','url']])
                st.write(f'Posts without mental health markers (total: {np.sum(~preds[:,1])}):')
                st.write(st.session_state.df_sample.iloc[:preds.shape[0],:].loc[~preds[:,1],['body','url']])


