import json
import random
import keras
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import helperFunctions
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

app = FastAPI(root_path="")

# uvicorn routerMain:app --reload

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading test set and preprocessing test set for input into the model

dataset = pd.read_csv('test_2.tsv', sep='\t', error_bad_lines=False)
pd.set_option('display.max_columns', None)
dataset.columns = ["tweet_id","username","timestamp","#followers","#retweets","#friends","#favorites","entities","sentiment","mentions","hashtags","urls"]
rawDataset = dataset.copy()
rawDataset.columns = ["tweet_id","username","timestamp","num_followers","num_retweets","num_friends","num_favorites","entities","sentiment","mentions","hashtags","urls"]

dataset["date"]=dataset["timestamp"].apply(lambda x: helperFunctions.createDate(x))
dataset["date"]=pd.to_datetime(dataset["date"],format="%d-%m-%Y %H:%M:%S")

covidKeywordsList = helperFunctions.importCovidKeywords("covid_keywords.txt")
stopWords = helperFunctions.importStopWords("stop_words.txt")
usernameDict = helperFunctions.importJson("usernameDict.json")
standardizeDict = helperFunctions.importJson("standardizeDict.json")

preprocessedDataset = helperFunctions.preprocessData(dataset, covidKeywordsList, stopWords, isTrainData=False, usernameDict=usernameDict, standardizeDict=standardizeDict)
copyOfPreprocessedDataset = preprocessedDataset.copy()
copyOfPreprocessedDataset.columns = ['tweet_id', 'username', 'timestamp', 'num_followers', 'num_retweets',
       'num_friends', 'num_favorites', 'entities', 'sentiment', 'mentions',
       'hashtags', 'urls', 'date', 'followers_null_ind', 'friends_null_ind',
       'entity_null', 'hashtags_null', 'urls_null', 'mentions_null',
       'keyword_hashtags', 'keyword_entities', 'sentiment_encoded',
       'num_entities', 'num_urls', 'num_hashtags', 'num_mentions',
       'unique_hashtags', 'hashtags_char', 'week', 'day', 'time', 'year',
       'followTofriends', 'friendsTofavorites', 'favoritesTofollow']

datasetEntries = preprocessedDataset.shape[0]

word_vec = np.loadtxt("./word_vec.txt", dtype=float)
vocab_len = 188823
tokenizer = helperFunctions.importTokenizer("tokenizer.pickle")
model = helperFunctions.importModel("weights_without_tweet_id.h5", vocab_len, word_vec, 200, usernameDict)

@app.get("/") 
async def root(): 
    return {"message": "Welcome to my API"}

@app.get("/getRandomTweet")
async def getRandomTweet():  
    random_number = random.randrange(1,datasetEntries)
    rawOutput = json.loads(rawDataset.loc[random_number].to_json(orient="index"))
    return {"raw":rawOutput, "index": random_number}

@app.get("/getTweetPredictionStatistics")
async def getTweetPredictionStatistics(index: int):
    stats = copyOfPreprocessedDataset.loc[index].to_json(orient="index")
    targetRow = preprocessedDataset.loc[index]
    batch = helperFunctions.processInput(targetRow, tokenizer, stopWords)
    prediction = model.predict(batch)
    output = prediction[0][0]
    return {"stats":json.loads(stats), "prediction": output.item()}

