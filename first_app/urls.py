from django.contrib import admin
from django.urls import path

from first_app import views


urlpatterns = [
        path("", views.homepage, name="plag_check"),
        path("thesis_upload_page/", views.thesis_upload_page, name="thesis_upload_page"),
        path("thesis_upload_succesfull/", views.thesis_upload_succesfull_page, name="thesis_upload"),
        path('chech_plagiarism/', views.plagiarism_check, name='extract_paragraphs'),
]