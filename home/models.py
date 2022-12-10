from django.db import models
from .letor import Letor

# Create your models here.
def eval_lambdamart(k=10, query = "six"):
    letor_model = Letor()
    letor_model.load()
    letor_model.load_lsi()
    res = []
    try:
        for doc in letor_model.predict_result(query, k):
            res.append(doc)
        return res
    except:
        return None
