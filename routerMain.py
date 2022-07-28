import json
import random
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd

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

trainDataset = pd.read_csv('C:/Users/Shawn/Github.com/AI_Project/TweetsCOV19.tsv', sep='\t', error_bad_lines=False)
trainDataset.columns = ["tweet_id","username","timestamp","num_followers","num_retweets","num_friends","num_favorites","entities","sentiment","mentions","hashtags","urls"]
datasetSize = trainDataset.shape[0]

@app.get("/") 
async def root(): 
    return {"message": "Welcome to my API"}

@app.get("/getRandomTweet")
async def getRandomTweet():  
    random_number = random.randrange(1,datasetSize)
    output = trainDataset.loc[random_number].to_json(orient="index")
    return json.loads(output)  

from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder


class tweetInput(BaseModel):
    tweet_id: int
    username: str
    timestamp: str
    num_followers: int
    num_retweets: int
    num_friends: int
    num_favorites: int
    entities: str
    sentiment: str
    mentions: str
    hashtags: str
    urls: str

@app.post("/getTweetPredictionStatistics")
async def getTweetPredictionStatistics(input: tweetInput):
    return {"tweet":"hello"}