from django.shortcuts import render, redirect
from movie.models import Member
import hashlib

def home(request):
    if 'userid' not in request.session.keys():
        return render(request, 'movie/login.html')
    else:
        return render(request, 'movie/main.html')

def login(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        passwd = request.POST['passwd']
        passwd = hashlib.sha256(passwd.encode()).hexdigest()
        row = Member.objects.filter(userid=userid, passwd=passwd)[0]
        if row is not None:
            request.session['userid'] = userid
            request.session['name'] = row.name
            return render(request, 'movie/main.html')
        else:
            return render(request, 'movie/login.html',
                          {'msg': '아이디 또는 비밀번호가 일치하지 않습니다.'})
    else:
        return render(request, 'movie/login.html')

def join(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        passwd = request.POST['passwd']
        passwd = hashlib.sha256(passwd.encode()).hexdigest()
        name = request.POST['name']
        address = request.POST['address']
        tel = request.POST['tel']
        Member(userid=userid, passwd=passwd, name=name, address=address, tel=tel).save()
        request.session['userid'] = userid
        request.session['name'] = name
        return render(request, 'movie/main.html')
    else:
        return render(request, 'movie/join.html')

def logout(request):
    request.session.clear()
    return redirect('/')
