from django import forms

class DocxUploadForm(forms.Form):
    docx_file = forms.FileField(label='Upload a DOCX file')
