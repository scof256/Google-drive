from django.db import models
from django.contrib.auth.models import User

YEAR_CHOICES = [
    ('Year 1', 'Year 1'),
    ('Year 2', 'Year 2'),
    ('Year 3', 'Year 3'),
    ('Year 4', 'Year 4'),
    ('Year 5', 'Year 5'),
    ('Year 6', 'Year 6'),
    ('General (N/A)', 'General (N/A)'),
]

class Folder(models.Model):
    foldername = models.CharField(max_length=50)
    folderdesc = models.CharField(max_length=255)
    folderuser = models.ForeignKey(User, on_delete=models.CASCADE)
    parent_folder = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='subfolders')

    def __str__(self):
        return self.foldername

class File(models.Model):
    filetitle = models.CharField(max_length=50)
    folder = models.ForeignKey(Folder, on_delete=models.CASCADE, related_name='files')
    file = models.FileField(upload_to="Files")
    year = models.CharField(max_length=20, choices=YEAR_CHOICES, default='General (N/A)')
    course = models.CharField(max_length=100)
    course_unit = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    likes = models.ManyToManyField(User, related_name='liked_files', blank=True)
    namespace = models.CharField(max_length=255, unique=True, null=True, blank=True)  # Allow NULL values

    

    def __str__(self):
        return self.filetitle


class Bookmark(models.Model):
    file = models.ForeignKey(File, on_delete=models.CASCADE, related_name='bookmarks')
    folder = models.ForeignKey(Folder, on_delete=models.CASCADE, related_name='bookmarks')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='bookmarks')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} bookmarked {self.file.filetitle} in {self.folder.foldername}"

class FileEmbedding(models.Model):
    file = models.ForeignKey(File, on_delete=models.CASCADE, related_name='embeddings')
    embedding = models.BinaryField()

    def __str__(self):
        return f"Embedding for {self.file.filetitle}"