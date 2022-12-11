from django.urls import path

from . import views

app_name = 'home'

urlpatterns = [
    path('', views.index, name='index'),
    path('clicked/<int:coll_ids>/<int:ids>', views.clicked, name='clicked')
]