from django.db import models
import pickle
import os
import sys
import lightgbm as lgb
from .bsbi import BSBIIndex
from .compression import VBEPostings
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
import numpy as np

# Create your models here.\
def vectorize(text, model):
        dictionary = Dictionary()
        NUM_LATENT_TOPICS = 200
        rep = [topic_value for (_, topic_value) in model[dictionary.doc2bow(text)]]
        return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

def features_processing(query, doc, model):
        v_q = vectorize(query, model)
        v_d = vectorize(doc, model)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist] 

def eval_lambdamart(k=10, query = "six"):
    directoire = os.path.dirname(__file__)
    rankModel = pickle.load(open(os.path.join(directoire, f'modelletor/{str("model3")}.pkl'), 'rb'))
    lsiModel = pickle.load(open(os.path.join(directoire, f'modelletor/{str("model_lsi1")}.pkl'), 'rb'))
    BSBI_instance = BSBIIndex(data_dir=os.path.join(directoire, 'collection'),
                              postings_encoding=VBEPostings,
                              output_dir=os.path.join(directoire,'index'))
    docs = []
    for (_, doc) in BSBI_instance.retrieve_0_5_tf_max_norm_smooth_idf(query, k):
        d = doc.replace("\\", "/").split("collection")[1][1:]
        docs.append(d)
    X_unseen = []
    if len(docs) < 1:
        return None
    else:
        docums = []
        for doc in docs:
            text = open(os.path.join(directoire, "collection/") + doc).read()
            docums.append(text.lower())
        for doc in docums:
            X_unseen.append(features_processing(query.split(), doc.split(), lsiModel))

        X_unseen = np.array(X_unseen)
        pred_score = rankModel.predict(X_unseen)
        scores = [x for x in zip([did for did in docs], pred_score)]
        sorted_did_scores = sorted(scores, key=lambda tup: tup[1], reverse=True)
        print([doc for doc,_ in sorted_did_scores])
        return [doc for doc,_ in sorted_did_scores]
    #     score = letor.predict_ranker(X_unseen)
    #     did_scores = [x for x in zip([did for (did, _) in docs], score)]
    #     sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

    #     return sorted_did_scores
    # X_unseen = []

    # docs = []

    # for (_, doc) in BSBI_instance.bm_25(query, k=k):
    #     text = open(os.path.join(this_dir, "collection/") + doc).read()
    #     text = text.lower()
    #     did = int(re.search(r'.*\/(.*)\.txt', doc).group(1))
    #     docs.append([did, text])

    # if len(docs) < 1:
    #     return None

    # else:
    #     for _, doc in docs:
    #         X_unseen.append(letor.features(query.split(), doc.split()))

    #     X_unseen = np.array(X_unseen)
    #     score = letor.predict_ranker(X_unseen)
    #     did_scores = [x for x in zip([did for (did, _) in docs], score)]
    #     sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

    #     return sorted_did_scores
    # letor_model = Letor()
    # letor_model.load()
    # letor_model.load_lsi()
    # res = []
    # try:
    #     for doc in letor_model.predict_result(query, k):
    #         res.append(doc)
    #     return res
    # except:
    #     return None
