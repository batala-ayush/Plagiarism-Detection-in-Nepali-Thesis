from django.contrib import admin
from django.urls import path

from first_app import views


urlpatterns = [
        path("", views.homepage, name="home_page"),
        #path("index/", views.homepage, name="home_page"),
        path("synset/", views.synset, name="synset"),
        path("upload_text/", views.upload_text, name="upload_text"),
        path("pos_tagger/", views.pos_tagger, name="pos_tagger"),
        path("thesis_upload_page/", views.thesis_upload_page, name="thesis_upload_page"),
        path("thesis_upload_succesfull_page/", views.thesis_upload_succesfull_page, name="thesis_upload_succesfull_page"),
        path('check_plagiarism/', views.plagiarism_check, name='extract_paragraphs'),
        #path('check_plagiarism_successful/', views.check_plagiarism_successful, name='check_plagiarism_successful'),
]