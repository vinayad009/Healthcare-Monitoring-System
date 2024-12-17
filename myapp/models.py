from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class UserDetail(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    gender = models.CharField(max_length=1, null=False, blank=False)
    age = models.IntegerField(null=True, blank=True)
    weight = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"{self.user.first_name}"
