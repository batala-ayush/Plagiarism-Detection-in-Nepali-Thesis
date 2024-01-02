from django.db import models

# Create your models here.

class thesis_docx(models.Model):
    thesis = models.FileField(upload_to='thesis_files') # Uploaded files will be stored in the 'thesis_files/' directory inside MEDIA_ROOT