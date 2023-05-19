from django.db import models

class Member(models.Model):
    userid = models.CharField(max_length=50, null=False, primary_key=True)
    passwd = models.CharField(max_length=500, null=False)
    name = models.CharField(max_length=20, null=False)
    address = models.CharField(max_length=20, null=False)
    tel = models.CharField(max_length=20, null=True)