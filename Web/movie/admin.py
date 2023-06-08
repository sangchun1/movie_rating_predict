from django.contrib import admin
from .models import movie

# Register your models here.
@admin.register(movie)
class PostAdmin(admin.ModelAdmin):
    pass