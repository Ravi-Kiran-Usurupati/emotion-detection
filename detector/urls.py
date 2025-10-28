from django.urls import path
from . import views

# This file maps URLs to views in the 'detector' app

urlpatterns = [
    # /
    # Maps the root URL of the app to the 'index' view
    path('', views.index, name='index'),
    
    # /predict/
    # This path must match the view function 'predict_emotion'
    # and its name must be 'predict_emotion' to match the template
    path('predict/', views.predict_emotion, name='predict_emotion'),
]
