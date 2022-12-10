import os
import pickle
import contextlib
import heapq
from pydoc import doc
import time
import math
from turtle import pos
import re
from .bsbi import BSBIIndex
from .compression import VBEPostings
import random
import lightgbm as lgb
import subprocess
import numpy as np

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
import nltk as nt

class Letor:
    # The init will retrieve the documents, the queries, and the qrel datas for training purpose. It also stores all of the variables used in this class
    def __init__(self):
        self.documents = {}
        self.queries = {}
        self.q_docs_rel = {}
        self.group_qid_count = []
        self.dataset = []
        self.dictionary = Dictionary()
        self.NUM_NEGATIVES = 1
        self.NUM_LATENT_TOPICS = 200
        self.directoire = os.path.dirname(__file__)
        self.BSBI_instance = BSBIIndex(data_dir=os.path.join(self.directoire, 'collection'),
                              postings_encoding=VBEPostings,
                              output_dir=os.path.join(self.directoire,'index'))
    
    # This function is used to clean the training data, stemming, tokenization, etc.
    def process_corp(self, corpus):
        stemmer = nt.stem.PorterStemmer()
        stopword = set(nt.corpus.stopwords.words('english'))
        tokenizer = nt.tokenize
        return [stemmer.stem(x) for x in [w for w in tokenizer.word_tokenize(re.sub("[^\w\s]", " ", re.sub("\s+", " ", corpus.lower()))) if not w.lower() in stopword]]

    # This is the vectorization function
    def vectorize(self, text, model, dictionary):
        rep = [topic_value for (_, topic_value) in model[dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    # This function will process the features of the data
    def features_processing(self, query, doc, model, dictionary):
        v_q = self.vectorize(query, model, dictionary)
        v_d = self.vectorize(doc, model, dictionary)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    # This function will generate the LSI model for the data (Assuming the datasets are alread available and downloaded)
    # Note : This model cannot be saved.
    def lsi(self):
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        self.Lsimodel = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS)
        X = []
        Y = []
        for (query, doc, rel) in self.dataset:
            X.append(self.features_processing(query, doc, self.Lsimodel, self.dictionary))
            Y.append(rel)
        self.X = np.array(X)
        self.Y = np.array(Y)
    
    def train(self, model = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)):
        self.rankModel = model
        self.rankModel.fit(self.X, self.Y, group = self.group_qid_count, verbose = 10)

    def train_without_lsi(self, model = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)):
        self.lsi()
        self.train(model)
    
    def predict(self, query, k=100):
        # We need to insert what model we're going to use first
        model = self.Lsimodel
        try:
            docs = [doc  for (scores, doc) in self.BSBI_instance.retrieve_0_5_tf_max_norm_smooth_idf(query, k)]
            local_docs = []
            for i in docs:
                doc_text = ""
                i = os.path.join(self.directoire, str(i)[2:])
                with open(i, "r", encoding='utf-8') as f:
                    doc_text = self.process_corp(f.read())
                local_docs.append((i, doc_text))
            X_unseen = []
            for _, doc in local_docs:
                X_unseen.append(self.features_processing(query.split(), doc, model, self.dictionary))
            X_unseen = np.array(X_unseen)
            predicted_score = self.rankModel.predict(X_unseen)
            did_scores = [x for x in zip([did for (did, _) in local_docs], predicted_score)]
            return sorted(did_scores, key = lambda tup: tup[1], reverse = True)
        except:
            return None

    def predict_result(self, query, k=100):
        datas = self.predict(query, k)
        return [doc for doc,_ in datas]


    def save(self, name="model3"):
        pickle.dump(self.rankModel, open(os.path.join(self.directoire, f'modelletor/{str(name)}.pkl'), 'wb'))

    def load(self, name="model3"):   
        self.rankModel = pickle.load(open(os.path.join(self.directoire, f'modelletor/{str(name)}.pkl'), 'rb'))

    def save_lsi(self, name="model_lsi1"):
        pickle.dump(self.Lsimodel, open(os.path.join(self.directoire, f'modelletor/{str(name)}.pkl'), 'wb'))

    def load_lsi(self, name="model_lsi1"):   
        self.Lsimodel = pickle.load(open(os.path.join(self.directoire, f'modelletor/{str(name)}.pkl'), 'rb'))
        # self.dictionary = Dictionary()
        # X = []
        # Y = []
        # for (query, doc, rel) in self.dataset:
        #     X.append(self.features_processing(query, doc, self.Lsimodel, self.dictionary))
        #     Y.append(rel)
        # self.X = np.array(X)
        # self.Y = np.array(Y)
        # alu

    def save_both(self, name):
        self.save(name)
        self.save_lsi(name)

    def load_both(self, name):
        self.load(name)
        self.load_lsi(name)



# Training and model saving is in trainmodel.py
# LSI training and saving is in lsitrain.py