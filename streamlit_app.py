import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
import tweepy # tweepy module to interact with Twitter
from tweepy import OAuthHandler # Used for authentication
from tweepy import Cursor # Used to perform pagination
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import datetime as dt
####
#packages for regression
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder 
st.title("Twitter Keyword Dashboard")
st.caption("Determining what words lead to more Likes and Retweets")
##input parameters
handle = st.text_input("Twitter Handle", value="@JoeBiden", max_chars= 100, placeholder= 'Enter a Twitter Handle', disabled=False)
handle = str.replace(handle, "@", "")

##############################
#get data

#Twitter Authentification Credentials
#Please update with your own credentials

cons_key = "zb9UbcG60lwbKXNMsaSS1kI3O" 
cons_secret = "gHueFvA0s6uiivq50Z7E47z2ApNNPFS1wkSF2zyb4BJKCdMrpY"
acc_token = '1510386735720677385-qEYcyId4jjXD1MLwsm3SiZKY3u4xI6'
acc_secret = "7ClAeBHNud1Ad4lIMK2DudGs6gnmlLKZk0DEqeHTbouwn"

# (1). Athentication Function
def get_twitter_auth():
    """
    @return:
        - the authentification to Twitter
    """
    try:
        consumer_key = cons_key
        consumer_secret = cons_secret
        access_token = acc_token
        access_secret = acc_secret
        
    except KeyError:
        sys.stderr.write("Twitter Environment Variable not Set\n")
        sys.exit(1)
        
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    return auth
# (2). Client function to access the authentication API
def get_twitter_client():
    """
    @return:
        - the client to access the authentification API
    """
    auth = get_twitter_auth()
    client = tweepy.API(auth, wait_on_rate_limit=True)
    return client
# (3). Function creating final dataframe
@st.experimental_memo
def get_tweets_from_user(twitter_user_name, page_limit=40, count_tweet=200):
    """
    @params:
        - twitter_user_name: the twitter username of a user (company, etc.)
        - page_limit: the total number of pages (max=16)
        - count_tweet: maximum number to be retrieved from a page
        
    @return
        - all the tweets from the user twitter_user_name
    """
    client = get_twitter_client()
    
    all_tweets = []
    
    for page in Cursor(client.user_timeline, 
                        screen_name=twitter_user_name, 
                        count=count_tweet).pages(page_limit):
        for tweet in page:
            parsed_tweet = {}
            parsed_tweet['date'] = tweet.created_at
            parsed_tweet['author'] = tweet.user.name
            parsed_tweet['twitter_name'] = tweet.user.screen_name
            parsed_tweet['text'] = tweet.text
            parsed_tweet['number_of_likes'] = tweet.favorite_count
            parsed_tweet['number_of_retweets'] = tweet.retweet_count
                
            all_tweets.append(parsed_tweet)
    
    # Create dataframe 
    df = pd.DataFrame(all_tweets)
    
    # Revome duplicates if there are any
    df = df.drop_duplicates( "text" , keep='first')
    
    return df

tweets_df = get_tweets_from_user(handle)
### prepare text for analysis
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer as stemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
import pandas as pd
import numpy as np

nltk.download('wordnet')
nltk.download('omw-1.4')
my_stop_words = STOPWORDS.union(set(['https', 'usda', 'thank']))
def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def string_preprocess(text):
    result = ''
    for token in gensim.utils.simple_preprocess(text):
        if token not in my_stop_words and len(token) > 3:
            result = result + lemmatize_stemming(token) + ' '
    return result
tweets_df['cleaned_text'] = tweets_df['text'].apply(lambda x: string_preprocess(x))

### regression analysis
from sklearn.linear_model import Ridge
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweets_df['cleaned_text'])

estimators = [("tf_idf", TfidfVectorizer()), 
              ("ridge", Ridge())]
model = Pipeline(estimators)
y = tweets_df['cleaned_text']
tweets_df['number_of_likes_and_retweets'] = tweets_df['number_of_likes'] + tweets_df['number_of_retweets']
#average tweet
tweet_avg = tweets_df['number_of_likes_and_retweets'].mean()
tweets_df['like_and_retweet_diff_from_average'] =  tweets_df['number_of_likes_and_retweets'] - tweet_avg

model.fit(tweets_df['cleaned_text'], tweets_df['like_and_retweet_diff_from_average'])
ridge_model = model.named_steps["ridge"]
tf_idf_model = model.named_steps["tf_idf"]
coefficients = pd.DataFrame({"keywords":tf_idf_model.get_feature_names(),
                             "impact score":ridge_model.coef_})

################################
#dashboard
charty = alt.Chart(coefficients.sort_values('impact score', ascending= False).head(20)).mark_bar().encode(
    x='impact score',
    y=alt.Y('keywords', sort='-x'))
st.text("{} tweets queiried".format(len(tweets_df)))
st.subheader('Top Keywords Assocated With More Likes and Retweets')
st.altair_chart(charty)
charty_last = alt.Chart(coefficients.sort_values('impact score', ascending= False).tail(20)).mark_bar().encode(
    x='impact score',
    y=alt.Y('keywords', sort='-x'))

st.subheader('Keywords Assocated With Less Likes and Retweets')
st.altair_chart(charty_last)
st.table(coefficients.sort_values("impact score", ascending=False).head(20))
###################
##footer
st.write("Data Sources: Twitter API")
st.text('By Sam Kobrin')


