from django.urls import path
from . import views

urlpatterns = [
    
    path('get_health', views.get_health, name='get_health'),
    path('nlp_prep', views.nlp_prep, name='nlp_prep'),
]