from django.db import models

class Mammo_Mass(models.Model):
    BI_RADS = models.IntegerField()
    Age = models.IntegerField()
    Shape = models.FloatField()
    Margin = models.IntegerField()
    Density = models.FloatField()
    Severity = models.IntegerField()

class Insurance(models.Model):
    age = models.IntegerField()
    bmi = models.FloatField()
    children = models.IntegerField()
    charges = models.FloatField()
    sex = models.IntegerField()
    smoker = models.IntegerField()
    region = models.IntegerField()