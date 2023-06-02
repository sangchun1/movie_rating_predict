from django.db import models
from django.utils import timezone



class movie(models.Model):
    id = models.CharField(max_length=50, null=False, primary_key=True, verbose_name='영화ID')
    title = models.CharField(max_length=50, verbose_name="영화제목")
    tot = models.CharField(max_length=50, verbose_name="누적 매출액")
    attendance = models.CharField(max_length=50, verbose_name="누적 관객수")
    screen = models.CharField(max_length=50, verbose_name="스크린 수")
    screening = models.CharField(max_length=50, verbose_name="상영횟수")
    year = models.IntegerField(verbose_name="개봉일")
    world = models.CharField(max_length=20, verbose_name="국가")
    genre = models.CharField(max_length=50, verbose_name="장르")
    star = models.IntegerField(verbose_name="별점")
    #review = models.TextField(verbose_name="리뷰내용")
    director = models.CharField(max_length=50, verbose_name="감독")
    actor = models.CharField(max_length=50, verbose_name="배우")
    running = models.IntegerField(verbose_name="영화상영시간")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

