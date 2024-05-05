from django import forms

class ObjectiveForm(forms.Form):
    objective = forms.CharField(max_length=500, widget=forms.Textarea(attrs={'placeholder': 'Enter your objective...'}))