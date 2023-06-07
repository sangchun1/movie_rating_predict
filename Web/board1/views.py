from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from  .models import Topic, Reply
from django.http import HttpRequest

def home(request):
    topics = Topic.objects.all()   #models의 Topic 개체 생성
    return render(request,'home.html',{'topics':topics})

