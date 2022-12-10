from django.shortcuts import render
from .models import eval_lambdamart
import os
import sys
# Create your views here.

def index(request):
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
        try:
            resultat_text = {}
            for i in resultat:
                with open(str(i), "r", encoding='utf-8') as filenya:
                    temp = str(i).split('collection\\')[1]
                    temp = temp.split("\\")[1]
                    resultat_text[str(temp)] = [filenya.read(), i]
            context = {
                'query': query,
                'textes': resultat_text,
                'loaded': 2
            }
            return render(request, 'home/index.html', context)
        except:
            context = {
                'query': query,
                'loaded': 1
            }
            return render(request, 'home/index.html', context)

def clicked(request, ids):
    returned = open(ids).read().lower()
    name = str(ids).split('collection\\')[1].split("\\")[1]
    context = {
        'name': name,
        'returned': returned,
    }
    return render(request, 'home/clicked.html', context)