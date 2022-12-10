from django.shortcuts import render

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
        context = {
            'query': query,
            'loaded': 1
        }
        return render(request, 'home/index.html', context)