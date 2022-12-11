from django.shortcuts import render
from .models import eval_lambdamart
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
        resultat = eval_lambdamart(100, query)
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
                resultat_text[str(doc)] = text
            #     docums.append(text.lower())
            # for i in docums:
            #     with open(i, "r", encoding='utf-8') as filenya:
            #         temp = str(i).split('collection\\')[1]
            #         temp = temp.split("\\")[1]
            #         resultat_text[str(temp)] = [filenya.read(), i]
            context = {
                'query': query,
                'textes': resultat_text,
                'loaded': 2
            }
            return render(request, 'home/index.html', context)
        # try:
        #     resultat_text = {}
        #     for i in resultat:
        #         with open(str(i), "r", encoding='utf-8') as filenya:
        #             temp = str(i).split('collection\\')[1]
        #             temp = temp.split("\\")[1]
        #             resultat_text[str(temp)] = [filenya.read(), i]
        #     context = {
        #         'query': query,
        #         'textes': resultat_text,
        #         'loaded': 2
        #     }
        #     return render(request, 'home/index.html', context)
        # except:
        #     context = {
        #         'query': query,
        #         'loaded': 1
        #     }
        #     return render(request, 'home/index.html', context)

def clicked(request, ids):
    directoire = os.path.dirname(__file__)
    name = ids
    text_dir = os.path.join(directoire, "collection/") + ids
    returned = open(text_dir).read()
    context = {
        'name': name,
        'returned': returned,
    }
    return render(request, 'home/clicked.html', context)