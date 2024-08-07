# Mental Health Monitoring on Social Media

## Background and Objective
Recent years have seen a rise in emotional distress among youths, prompting many to seek support on online forums and social media platforms. A text classification machine learning program can help identify signs of distress in these posts, enabling timely intervention from counselors or healthcare professionals. This project explores and compares various machine learning techniques for this text classification task. Models considered include Naive Bayes, KNN, Logistic Regression, and XGBoost, combined with vectorization methods like Count Vectorizers and Word2Vec.

## Approach
Text posts from a subreddit on mental health [r\mentalhealth](https://www.reddit.com/r/mentalhealth/) and on casual conversations [r\casualconversation](https://www.reddit.com/r/CasualConversation/) are used for model training with the former representing posts showing signs of emotional distress, and the latter for posts without such signs. The PRAW package is used to retrieve these posts via the reddit API.

## Dataset
The API retrieved dataset for the two classes are saved in 'subreddit_casualconversation_20240614.csv' and 'subreddit_mentalhealth_20240614.csv', which are combined and pre-processed to give the `df_raw` dataframe below. `df_raw` is to be further transformed for the corresponding machine learning models.

| Feature  | Type | Description                                                                                                  |
|----------|------|--------------------------------------------------------------------------------------------------------------|
| id       | obj  | Unique ID identifying test reddit post                                                                       |
| title    | obj  | Title of reddit post                                                                                         |
| body     | obj  | Main text of reddit post                                                                                     |
| comments | obj  | Comments and subcomments to reddit post                                                                      |
| target   | obj  | Content type of text as target for text classification system, either 'mentalhealth' or 'casualconversation' |
| url      | obj  | URL of original reddit post                                                                                  |


## Findings
The F1-scores of the different models used are show in the bar chart below. For the current case of identifying text posts with signs of psychological distress, the  model comparison here does not indicate significant differences in performance. That said, a relatively small dataset of approximately 1,000 datapoints in each category was used for model training. Furthermore, relatively little pre-processing has been done, beyond the application of generic vectorizers, stemming, and stop word removals. Therefore, there is likely room for improvement for each model, depending on the use case (e.g. flagging student text posts for referral to university counselors for further action), the predictive power required (e.g. precision, recall, etc), and the computational power available for inference. 


<img src=".\assets\f1_scores.png" alt="F1-scores of Various Models" width="700"/>


While the Multinomial Naive Bayes model trained on TF-IDF vectorized data appears to be only slightly inferior to the BERT model (0.95 vs 0.98 in F1 score), BERT is more certain about its predictions. This is shown in the distribution of predicted probabilities for mental health distress in a text post for the two models below. As shown, BERT has a much higher peak near 1.0 and fewer data points near the threshold, unlike the Multinomial NB model. This means that there is less ambiguity and uncertainty associated BERT predictions.

<img src=".\assets\probability_distribution_bert_mnb.png" alt="Predicted Probabilities of Mental Health Distress by BERT and MNB" width="700"/>

For further refinement, one could improvement performance by gathering more data as the current dataset is relatively modest for an NLP project. Additionally, further data pre-processing will certainly be advantageous. For example, emojis are frequently used in social media and can be properly transformed to extract better contextual information. Furthermore, the texts may contain weblinks, personal sign off messages, bot-generated messages, foreign (non-English in the current context) characters, etc, which may reduce the quality of the dataset. If computational resources are adequate for deep learning approaches, one could also experiment with different pre-trained BERT models, or other architectures altogether like GPT.

## App Description
 A simple app is created on streamlit to illustrate deployment possibilities. This app can be found in the './app' directory and can be launched by running `streamlit run Post_Classifier.py`. A similar methodology can be employed for other social media platforms or online forums.
 
 The app supports direct downloads from the social media site (Reddit here) for classification...

<img src=".\assets\multitext1.png" alt="Multi-text classificaiton: data source options" width="500"/>

<img src=".\assets\multitext2.png" alt="Multi-text classificaiton: illustration" width="500"/>

... or a single text classification, providing probability of mental health distress detected in the supplied text.

<img src=".\assets\singletext.png" alt="Multi-text classificaiton: data source options" width="500"/>





