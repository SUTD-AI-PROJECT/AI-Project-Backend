import json
import pandas as pd
import gc
import numpy as np
import keras
import tensorflow as tf
import pickle
from keras.models import load_model
from tensorflow.keras.utils import Sequence
import re

def createDict(unique_val, start):
  count=start
  dic ={}
  for k in unique_val:
      dic[k]=count
      count+=1
  return dic

def preprocessData(Dataset, covidKeywords, stopWords, isTrainData, usernameDict, standardizeDict):
  #===========================================================================================================================
  # Preprocessing username
  if isTrainData:
    # If is train dataset, create the dictionary for later use
    # Filtering out people that have <tweets_threshold number of tweets
    tweetsThreshold = 10 
    userCounts = Dataset["username"].value_counts()
    userCounts = userCounts[userCounts>=tweetsThreshold] 

    # Creating dictionary of username mapping for later use in validation set
    usernameDict["username"] = createDict(userCounts.index, 1) 
    Dataset["username"] = Dataset["username"].map(usernameDict["username"])
    Dataset["username"] = Dataset["username"].fillna(0)

  else: 
    # If is not train dataset(i.e. is Validation or test set), use previously made username dictionary to map usernames
    # Mapping username column to username dictionary previously created when preprocessing trainDataset
    Dataset["username"] = Dataset["username"].map(usernameDict)
    Dataset["username"] = Dataset["username"].fillna(0)

  #===========================================================================================================================
  # Preprocessing null fields
  Dataset["followers_null_ind"] = Dataset["#followers"].isnull().astype(int)
  Dataset["friends_null_ind"] = Dataset["#friends"].isnull().astype(int)

  Dataset["entity_null"] = (Dataset["entities"]=="null;").astype(int)
  Dataset["hashtags_null"] = (Dataset["hashtags"]=="null;").astype(int)
  Dataset["urls_null"] = (Dataset["urls"]=="null;").astype(int)
  Dataset["mentions_null"] = (Dataset["mentions"]=="null;").astype(int)

  #===========================================================================================================================
  # Preprocessing hashtags
  def keyword_hashtags(x,key_covid):
    if x=="null;":
        return 0
    else:
      # print(x)
      s = str(x).split(" ")
      ff=[]
      for z in s:
          ff.append(z.lower())
      count=0
      for zz in ff:
          if zz in key_covid:
              count+=1
      return count
        
  Dataset["keyword_hashtags"] = Dataset["hashtags"].apply(lambda x: keyword_hashtags(x, covidKeywords))

  #===========================================================================================================================
  # Preprocessing entities
  def keyword_entities(x,key_covid):
    if x=="null;":
        return 0
    else:
      # print(x)
      s = str(x).split(";")
      ff=[]
      for z in s[:-1]:
          temp = z.split(":")
          ff.append(temp[0].lower())
          ff.append(temp[1].lower())
      count=0
      for zz in ff:
          if zz in key_covid:
              count+=1
      return count

  Dataset["keyword_entities"] = Dataset["entities"].apply(lambda x: keyword_entities(x, covidKeywords))

  #===========================================================================================================================
  # Preprocessing sentiments
  def one_hot_sentiment(x):
      spl = x.split(" ")
      d =[0]*10
      d[int(spl[0])-1]=1
      d[int(spl[-1])]=1
      return d

  Dataset["sentiment_encoded"] = Dataset["sentiment"].apply(lambda x: one_hot_sentiment(x))

  #===========================================================================================================================
  # Preprocessing counts
  def count(x,sep):
    if x!="null;" and str(x)!="nan":
        cc = x.split(sep)
        return len(cc)
    else:
        return 0
    
  def unique_hashtags(x):
      if x=="null;" or str(x)=="nan":
          return 0
      else:
          x=x.split(" ")
          return np.unique(x).size/len(x)
      
  def count_words(x):
      if x=="null;" or str(x)=="nan":
          return 0
      else:
          return len(x)

  Dataset["no_entities"] = Dataset["entities"].apply(lambda x: count(x,";"))
  Dataset["no_urls"] = Dataset["urls"].apply(lambda x: count(x,":-:"))
  Dataset["no_hashtags"] = Dataset["hashtags"].apply(lambda x: count(x," "))
  Dataset["no_mentions"] = Dataset["mentions"].apply(lambda x: count(x," "))
  Dataset["unique_hashtags"] = Dataset["hashtags"].apply(lambda x: unique_hashtags(x))
  Dataset["hashtags_char"] = Dataset["hashtags"].apply(lambda x: count_words(x))

  #===========================================================================================================================
  # Preprocessing timestamp
  def one_hot_week(x,dict_):
    len_=len(dict_)
    z=[0]*len_
    z[dict_[x]]=1
    return z
    
  def conv_dtime(v):
        v = v.split(":")
        return (float(v[0])*3600+float(v[1])*60+float(v[2]))/3600

  Dataset["week"] = Dataset["timestamp"].apply(lambda x: x.split(" ")[0])
  #Dataset["month"] = Dataset["timestamp"].apply(lambda x: x.split(" ")[1])
  Dataset["day"] = Dataset["timestamp"].apply(lambda x: x.split(" ")[2])
  Dataset["time"] = Dataset["timestamp"].apply(lambda x: x.split(" ")[3])
  Dataset["year"] = Dataset["timestamp"].apply(lambda x: x.split(" ")[5])

  if isTrainData:
        usernameDict["week"] = createDict(Dataset["week"].unique(), 0)

  Dataset["week"] = Dataset["week"].apply(lambda x: one_hot_week(x,usernameDict["week"]))
  #Dataset["month"] = Dataset["month"].map(month_dict)
  Dataset["time"] = Dataset["time"].apply(lambda x: conv_dtime(x))
  Dataset["day"] = Dataset["day"].astype(int)
  Dataset["year"] = Dataset["year"].map({"2019":0,"2020":1})

  #===========================================================================================================================
  # Preprocessing ratio
  Dataset["follow/friends"]=Dataset["#followers"].astype(float)/(Dataset["#friends"].astype(float)+1)
  Dataset["friends/favorites"]=Dataset["#friends"].astype(float)/(Dataset["#favorites"].astype(float)+1)
  Dataset["favorites/follow"]=Dataset["#favorites"].astype(float)/(Dataset["#followers"].astype(float)+1)

  #===========================================================================================================================
  # log
  Dataset["#followers"]=np.log(Dataset["#followers"].astype(int)+1)
  Dataset["#friends"]=np.log(Dataset["#friends"].astype(int)+1)
  Dataset["#favorites"]=np.log(Dataset["#favorites"].astype(int)+1)
  Dataset["no_entities"]=np.log(Dataset["no_entities"].astype(int)+1)
  Dataset["no_urls"]=np.log(Dataset["no_urls"].astype(int)+1)
  Dataset["no_mentions"]=np.log(Dataset["no_mentions"].astype(int)+1)
  Dataset["no_hashtags"]=np.log(Dataset.no_hashtags+1)

  Dataset["day"]=np.log(Dataset["day"].astype(int)+1)
  Dataset["time"]=np.log(Dataset["time"].astype(int)+1)
  Dataset["follow/friends"]=np.log(Dataset["follow/friends"]+1)
  Dataset["friends/favorites"]=np.log(Dataset["friends/favorites"]+1)
  Dataset["favorites/follow"]=np.log(Dataset["favorites/follow"]+1)
  Dataset["unique_hashtags"]=np.log(Dataset["follow/friends"]+1)
  Dataset["hashtags_char"]=np.log(Dataset["hashtags_char"]+1)
  Dataset["keyword_entities"]=np.log(Dataset["keyword_entities"]+1)
  Dataset["keyword_hashtags"]=np.log(Dataset["keyword_hashtags"]+1)

  #===========================================================================================================================
  # Standardizing values = finding how many standard deviation away from mean
  featuresToStandardize=['#favorites', '#followers', '#friends', 'day', 'no_entities', 'no_hashtags', 
                        'no_mentions', 'no_urls','time',"follow/friends","friends/favorites","favorites/follow",
                      "unique_hashtags","hashtags_char","keyword_entities", "keyword_hashtags"]
  if isTrainData: 
    # If is train dataset, create the dictionary to save mean and standard deviation for later use in preprocessing validation set
    standardizeDict = {}
    for feature in featuresToStandardize:
      meanValue = Dataset[feature].mean()
      standardDeviationValue = Dataset[feature].std()
      Dataset[feature] = (Dataset[feature]-meanValue)/standardDeviationValue 
      standardizeDict[feature] = {"meanValue": meanValue, "standardDeviationValue": standardDeviationValue}
  else: 
    # If is not train dataset(i.e. is Validation or test set), use previously made mean and standard deviation dictionary from train dataset to standardize
    for feature in featuresToStandardize:
      Dataset[feature] = (Dataset[feature]-standardizeDict[feature]["meanValue"])/standardizeDict[feature]["standardDeviationValue"]

  #===========================================================================================================================
  # Returning relevant information
  if isTrainData: 
    # If is train dataset, return the preprocessed dataset, usernameDict and standardizeDict for use in validation and test dataset
    return Dataset, usernameDict, standardizeDict
  else: 
    # If is not train dataset(i.e. is Validation or test set), just return the preprocessed dataset
    return Dataset
  

monthStringToInt={"Jan":'1',"Feb":'2',"Mar":'3',"Apr":'4',"May":'5',"Jun":'6',"Sep":'9',"Oct":'10',"Nov":'11',"Dec":'12'}
def createDate(timestamp):
  lst = timestamp.split(" ")
  dt=lst[2]+"-"+monthStringToInt[lst[1]]+"-"+lst[-1]+" "+lst[3]
  return dt

  
def importCovidKeywords(path):
  covidKeywordsFile = open(f"{path}")
  covidKeywords = covidKeywordsFile.readlines()
  covidKeywordsList=[]
  for k in covidKeywords:
      covidKeywordsList+=k.split()
  covidKeywordsSet = set(covidKeywordsList)
  return covidKeywordsSet

def importStopWords(path):
  stopWordsFile = open(f"{path}")
  stopWords = stopWordsFile.readlines()
  stopWordsList=[]
  for k in stopWords:
      stopWordsList+=k.split()
  stopWordsSet = set(stopWordsList)
  return stopWordsSet


def importJson(path):
  with open(f"{path}") as json_file:
    data = json.load(json_file)
    # Print the type of data variable
    return data

def combine_entity(x):
    a = split_word(x,0)+" "+bigrams(x,0)
    a = " ".join(a.split())
    return a

def hashtag(k):
    if k=="null;":
        return ""
    else:
        k=str(k)
        orig=k.lower().split()
        # print(k)
        k= re.sub(r'[^a-zA-Z]'," ",k)
        # print(k)
        k=" ".join(k.split())
        # print(k)
        k=" ".join([a for a in re.split('([A-Z][a-z]+)', k) if a])
        # print(k)
        k=k.lower().split()+orig
        # print(k)
        ff=[]
        check_set=set()
        for zz in k:
            if zz not in check_set:
                ff.append(zz)
                check_set.add(zz)
        return " ".join(ff)

def process_urlPath(x,pos,stop_set):
    if x=="null;":
        return ""
    else:
        x = str(x)
        spl=x.split(":-:")[:-1]
        res=[]
        for k in spl:
            uert = split_url(k,"d")
            rt = re.sub("[^A-Za-z]"," ",uert[pos])
            rt = rt.split()
            ur_spl=[]
            for ul in rt:
                if ul not in stop_set:
                    ur_spl.append(ul)
            res.append(" ".join(ur_spl))
        return " ".join(res)

def split_url(line, part):
    # this is copy of split_url function from the URLNetrepository: https://github.com/Antimalweb/URLNet/blob/master/utils.py
    if line.startswith("http://"):
        line=line[7:]
    if line.startswith("https://"):
        line=line[8:]
    if line.startswith("ftp://"):
        line=line[6:]
    if line.startswith("www."):
        line = line[4:]
    slash_pos = line.find('/')
    if slash_pos > 0 and slash_pos < len(line)-1: # line = "fsdfsdf/sdfsdfsd"
        primarydomain = line[:slash_pos]
        path_argument = line[slash_pos+1:]
        path_argument_tokens = path_argument.split('/')
        pathtoken = "/".join(path_argument_tokens[:-1])
        last_pathtoken = path_argument_tokens[-1]
        if len(path_argument_tokens) > 2 and last_pathtoken == '':
            pathtoken = "/".join(path_argument_tokens[:-2])
            last_pathtoken = path_argument_tokens[-2]
        question_pos = last_pathtoken.find('?')
        if question_pos != -1:
            argument = last_pathtoken[question_pos+1:]
            pathtoken = pathtoken + "/" + last_pathtoken[:question_pos]     
        else:
            argument = ""
            pathtoken = pathtoken + "/" + last_pathtoken          
        last_slash_pos = pathtoken.rfind('/')
        sub_dir = pathtoken[:last_slash_pos]
        filename = pathtoken[last_slash_pos+1:]
        file_last_dot_pos = filename.rfind('.')
        if file_last_dot_pos != -1:
            file_extension = filename[file_last_dot_pos+1:]
            filename = filename[:file_last_dot_pos]
        else:
            file_extension = "" 
    elif slash_pos == 0:    # line = "/fsdfsdfsdfsdfsd"
        primarydomain = line[1:]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    elif slash_pos == len(line)-1:   # line = "fsdfsdfsdfsdfsd/"
        primarydomain = line[:-1]
        pathtoken = ""
        argument = ""
        sub_dir = ""     
        filename = ""
        file_extension = ""
    else:      # line = "fsdfsdfsdfsdfsd"
        primarydomain = line
        pathtoken = ""
        argument = ""
        sub_dir = "" 
        filename = ""
        file_extension = ""
    if part == 'pd':
        return primarydomain
    elif part == 'path':
        return pathtoken
    elif part == 'argument': 
        return argument 
    elif part == 'sub_dir': 
        return sub_dir 
    elif part == 'filename': 
        return filename 
    elif part == 'fe': 
        return file_extension
    elif part == 'others': 
        if len(argument) > 0: 
            return pathtoken + '?' +  argument 
        else: 
            return pathtoken 
    else:
        return primarydomain, pathtoken, argument, sub_dir, filename, file_extension

def split_word(x,pos):
    if x == "null;":
        return ""
    else:
        x = str(x)
        s = x.split(";")[:-1]
        ff=[]
        check_set=set()
        for z in s:
            d = z.split(":")
            ent=d[pos]
            ent2 = re.sub("%\d+"," ",d[pos+1])
            ent2 = re.sub("_","",ent2)
            ent2 = ent2.lower().split()
            #ent = [ent.lower()]+ent2
            ent = [ent.lower()]
            for zz in ent:
                if zz not in check_set:
                    ff.append(zz)
                    check_set.add(zz)
        rt= " ".join(ff)
        return rt

def create_bigram(h):
    dr=[]
    start=h[0]
    for k in h[1:]:
        dr.append(start+k)
        start=k
    return dr

def bigrams(x,pos):
    if x == "null;":
        return ""
    else:
        x = str(x)
        s = x.split(";")[:-1]
        ff=[]
        check_set=set()
        for z in s:
            d = z.split(":")
            ent2 = re.sub("%\d+"," ",d[pos+1])
            ent2 = re.sub("_"," ",ent2)
            ent2 = ent2.lower().split()
            if len(ent2)>1:
                ff+=create_bigram(ent2)
        if len(ff)==0:
            return ""
        else:
            return " ".join(ff)

def load_vectors(fname,count_words):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # n, d = map(int, fin.readline().split())
        data = {}
        data_list=[]
        for line in fin:
            tokens = line.rstrip().split(' ')
            tk = tokens[0]
            if tk in count_words:
                vec=list(map(float, tokens[1:]))
                data[tk] = vec
                data_list.append(vec)
        return data,data_list
    
    
def glove_load_vectors(fname,count_words):
        data={}
        fastvec = open(fname)
        counter=1
        data_list=[]
        while counter>0:
            try:
                f=fastvec.__next__()
                tokens = f.rstrip().split(' ')
                tk=tokens[0]
                if tk in count_words:
                    vec = list(map(float, tokens[1:]))
                    data[tk] = vec
                    data_list.append(vec)
                counter+=1
            except:
                print("total tokens",counter)
                counter=0
                pass
        return data,data_list

def create_embeddings(train_data,embedding_path,wordvec_name,stop_set,word_dim):

    entity1 =  train_data["entities"].apply(lambda x: combine_entity(x))
    mention_dt =  train_data["hashtags"].apply(lambda x: hashtag(x))
    url_dt1 =  train_data["urls"].apply(lambda x: process_urlPath(x,0,stop_set))
    url_dt2 =  train_data["urls"].apply(lambda x: process_urlPath(x,1,stop_set))
    mention_splt = train_data["mentions"].apply(lambda x: hashtag(x))
    
    dt_concat =pd.concat([entity1,mention_dt,url_dt1,url_dt2,mention_splt],axis=0)
    
    print("create entity tokenizer")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None)

    #tokenizer.fit_on_texts(pd.concat([entity1,mention_dt,url_dt,mention_splt],axis=0))
    tokenizer.fit_on_texts(dt_concat)
    
    count_thres = 15
    count_words = {w:c for w,c in tokenizer.word_counts.items() if c >= count_thres}

    word_counts= len(count_words)+1#one for oov and one for less count words

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=word_counts,
        filters='',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None)

    #tokenizer.fit_on_texts(pd.concat([entity1,mention_dt,url_dt,mention_splt],axis=0))
    tokenizer.fit_on_texts(dt_concat)
    
    print("load embedding vectors")
    if wordvec_name.split(".")[0]=="glove":
        fastvec,fastvec_list = glove_load_vectors(embedding_path,count_words)
    else:
        fastvec,fastvec_list = load_vectors(embedding_path,count_words)

    cand=np.array(fastvec_list,dtype='float32')
    mu=np.mean(cand, axis=0)
    Sigma=np.cov(cand.T)
    norm=np.random.multivariate_normal(mu, Sigma, 1)
    norm = list(np.reshape(norm, word_dim))

    word_counts = len(count_words)+1
    word_vectors = np.zeros((word_counts,word_dim))
    id_w = tokenizer.index_word

    for k in range(1,word_vectors.shape[0]):
        ky = id_w[k]
        if ky in fastvec:
            word_vectors[k,:]=fastvec[ky]
        else:
            word_vectors[k,:]= norm
    
    return tokenizer,word_counts,word_vectors

def attention_user(embed_entity,embed_user,cnn_option):
    user_embedding_word= tf.keras.layers.Dense(100,activation='relu')(embed_user)
    user_embedding_word= tf.keras.layers.Flatten()(user_embedding_word)
    if cnn_option==1:
        embed_entity = tf.keras.layers.Convolution1D(filters=150, kernel_size=3,  padding='same', activation='relu', strides=1)(embed_entity)
    else:
        embed_entity=tf.keras.layers.LSTM(150,return_sequences=True)(embed_entity)
    attention_a=tf.keras.layers.Dot((2, 1))([embed_entity,tf.keras.layers.Dense(150,activation='tanh')(user_embedding_word)])
    attention_weight = tf.keras.layers.Activation('softmax')(attention_a)
    news_rep=tf.keras.layers.Dot((1, 1))([embed_entity, attention_weight])
    return news_rep
  
  #custom loss metrics to track performance
def msle_function(actual,prediction):
    prediction = tf.keras.backend.exp(prediction)-1
    pred = tf.keras.backend.round(prediction)
    pred = tf.keras.backend.log(prediction+1)
    error = (actual-prediction)**2
    mean_error = tf.keras.backend.mean(error)
    return mean_error

def model(feature_dicts,word_vec,vec_d,vocab_len,cnn_option):
    user_length = len(feature_dicts["username"])+1
    #user embedding length
    embedding_len = 64
    tf.compat.v1.disable_eager_execution()
    #create input heads
    user_inp = tf.keras.layers.Input((1,))                 ##### might need to edit these numbers according to our choice of input
    sentiment_inp = tf.keras.layers.Input((10,))
    week_inp=tf.keras.layers.Input((7,))
    all_feats_inp = tf.keras.layers.Input((23,))
    entity_inp1 = tf.keras.layers.Input((10,))
    hashtag_inp = tf.keras.layers.Input((5,))
    urlPath_inp1 = tf.keras.layers.Input((3,))    #find out if this is host and path and change name accordingly
    urlPath_inp2 = tf.keras.layers.Input((15,))
    mentionsplt_inp = tf.keras.layers.Input((5,))

    #create embedding for users
    user_embed = tf.keras.layers.Embedding(input_dim=user_length, output_dim=embedding_len,embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=123),input_length=1)
    entity_embed = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=vec_d,weights=[word_vec],trainable=True)


    #query embeddings
    embed_user =user_embed(user_inp)
    embedding_user = tf.keras.layers.Lambda(lambda y: tf.keras.backend.squeeze(y, 1))(embed_user)

    embed_entity =entity_embed(entity_inp1)
    embed_entity= tf.keras.layers.Dropout(0.25)(embed_entity)
    entity_features_conv1 = attention_user(embed_entity,embed_user,cnn_option)


    embedding_hashtag =entity_embed(hashtag_inp)
    embedding_hashtag= tf.keras.layers.Dropout(0.25)(embedding_hashtag)
    hashtag_features_conv = attention_user(embedding_hashtag,embed_user,cnn_option)

    embedding_urlPath1 =entity_embed(urlPath_inp1)
    embedding_urlPath1= tf.keras.layers.Dropout(0.25)(embedding_urlPath1)
    urlPath_features_conv1 = attention_user(embedding_urlPath1,embed_user,cnn_option)

    embedding_urlPath2 =entity_embed(urlPath_inp2)
    embedding_urlPath2= tf.keras.layers.Dropout(0.25)(embedding_urlPath2)
    urlPath_features_conv2 = attention_user(embedding_urlPath2,embed_user,cnn_option)

    embedding_mentionsplt =entity_embed(mentionsplt_inp)
    embedding_mentionsplt= tf.keras.layers.Dropout(0.25)(embedding_mentionsplt)
    mentionsplt_features_conv = attention_user(embedding_mentionsplt,embed_user,cnn_option)


    inp_feats = tf.keras.layers.concatenate([embedding_user, sentiment_inp,week_inp,all_feats_inp,entity_features_conv1,hashtag_features_conv,urlPath_features_conv1,urlPath_features_conv2,mentionsplt_features_conv], 1)

    fc1 = tf.keras.layers.Dense(500, activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(inp_feats)
    drop1 = tf.keras.layers.Dropout(rate=0.15,seed=456)(fc1)
    fc2 = tf.keras.layers.Dense(150, activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=456))(drop1)
    drop2 = tf.keras.layers.Dropout(rate=0.15,seed=34)(fc2)
    out = tf.keras.layers.Dense(1,activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=789))(drop2)
    
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model_cp = tf.keras.models.Model(inputs=[user_inp,sentiment_inp,week_inp,all_feats_inp,entity_inp1,hashtag_inp,urlPath_inp1,urlPath_inp2,mentionsplt_inp],outputs=[out])
    # model_cp.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(),optimizer=optimizer,metrics=[msle_function,'accuracy'])## edited mean squared error to meansquared log loss
    model_cp.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(),optimizer=optimizer,metrics=["mean_squared_logarithmic_error", "accuracy"])
    return model_cp

def importModel(weightsPath, vocab_len, word_vec, wordvec_dim, usernameDict):
  model_cp = model(usernameDict,word_vec,wordvec_dim,vocab_len,cnn_option=1)
  model_cp.load_weights(weightsPath)
  return model_cp

def importTokenizer(path):
  with open(path, 'rb') as handle:
    tokenizer = pickle.load(handle)
  return tokenizer

def processInput(dfRow, tokenizer, stop_set):
    username = np.array([dfRow["username"]])
    print(dfRow["sentiment_encoded"])
    print(type(dfRow["sentiment_encoded"]))
    sentiment_encode = np.array([np.array(dfRow["sentiment_encoded"])])
    week_encode = np.array([np.array(dfRow["week"])])
    other_f = ['#favorites', '#followers', '#friends', 'day',
        'no_entities', 'no_hashtags', 'no_mentions', 'no_urls','time',"year","follow/friends","friends/favorites","favorites/follow","unique_hashtags","hashtags_char","entity_null","hashtags_null","urls_null","mentions_null","keyword_entities","keyword_hashtags",
                    'followers_null_ind', 'friends_null_ind']
    all_fea= np.array([np.array(dfRow[other_f].values)])

    entity1 =  [combine_entity(dfRow["entities"])]
    # dfRow["entities"].apply(lambda x: combine_entity(x))
    entity_sequences1 = tokenizer.texts_to_sequences(entity1)
    entity_pad1 = tf.keras.preprocessing.sequence.pad_sequences(entity_sequences1, maxlen=10, dtype='int32', padding='pre', truncating='post')

    hashtag_process =  [hashtag(dfRow["hashtags"])]
    # dfRow["hashtags"].apply(lambda x: hashtag(x))
    valid_hashtag = tokenizer.texts_to_sequences(hashtag_process)
    hashtag_valid = tf.keras.preprocessing.sequence.pad_sequences(valid_hashtag, maxlen=5, dtype='int32', padding='pre', truncating='post')

    url_dt1 =  [process_urlPath(dfRow["urls"], 0, stop_set)]
    # dfRow["urls"].apply(lambda x: process_urlPath(x,0,self.stop_set))
    urlPath_sequences1 = tokenizer.texts_to_sequences(url_dt1)
    urlPath_valid1 = tf.keras.preprocessing.sequence.pad_sequences(urlPath_sequences1, maxlen=3, dtype='int32', padding='pre', truncating='post')

    url_dt2 =  [process_urlPath(dfRow["urls"], 1, stop_set)]
    urlPath_sequences2 = tokenizer.texts_to_sequences(url_dt2)
    urlPath_valid2 = tf.keras.preprocessing.sequence.pad_sequences(urlPath_sequences2, maxlen=15, dtype='int32', padding='pre', truncating='post')

    mention_splt =  [hashtag(dfRow["mentions"])]
    # dfRow["mentions"].apply(lambda x: hashtag(x))
    mention_validsplt = tokenizer.texts_to_sequences(mention_splt)
    mention_validsplt = tf.keras.preprocessing.sequence.pad_sequences(mention_validsplt, maxlen=5, dtype='int32', padding='pre', truncating='post')



    batch_x = [username,sentiment_encode,week_encode,all_fea,entity_pad1,hashtag_valid,urlPath_valid1,urlPath_valid2,mention_validsplt]
    return batch_x