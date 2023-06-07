from django.http import HttpResponse
from django.shortcuts import render

from .models import movie


def index(request):
    movies = movie.objects.all()
    context = {
        "movies": movies
    }
    return render(request,'movie.html', context=context)


