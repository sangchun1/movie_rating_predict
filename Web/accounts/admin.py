from django.contrib import admin
from movie.models import Member

class MovieAdmin(admin.ModelAdmin):
    list_display = ("userid", "name", "address", "tel")

admin.site.register(Member, MovieAdmin)

