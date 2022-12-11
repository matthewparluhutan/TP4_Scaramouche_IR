from django.shortcuts import render
from .models import eval_lambdamart
from datetime import datetime
import os
import sys
# Create your views here.

def index(request):
    directoire = os.path.dirname(__file__)
    query = request.GET.get('searchInput')
    if query == None or query == "":
        context = {
            'query': query,
            'loaded': 0
        }
        return render(request, 'home/index.html', context)
    else:
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        start_time = datetime.now()
        resultat = eval_lambdamart(100, query)
        end_time = datetime.now()
        timenya = str(end_time - start_time).split(":")[-1]

        if resultat is None:
            context = {
                'query': query,
                'loaded': 1
            }
            return render(request, 'home/index.html', context)
        else:
            resultat_text = {}
            for doc in resultat:
                text_dir = os.path.join(directoire, "collection/") + doc
                text = open(text_dir).read()
                coll_ids = int((doc).split("/")[0])
                ids = int((doc).split("/")[1].split(".")[0])
                resultat_text[str(doc)] = [text, coll_ids, ids]
            context = {
                'query': query,
                'textes': resultat_text,
                'loaded': 2,
                'time': timenya,
            }
            return render(request, 'home/index.html', context)

def clicked(request, coll_ids, ids):
    directoire = os.path.dirname(__file__)
    data_ids = f"{str(coll_ids)}/{str(ids)}.txt"
    text_dir = os.path.join(directoire, "collection/") + data_ids
    returned = open(text_dir).read()

    context = {
        'name': f"Document {str(ids)}",
        'returned': returned,
    }
    return render(request, 'home/clicked.html', context)