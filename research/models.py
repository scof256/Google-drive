from django.db import models

# Create your models here.
# models.py
from django.db import models

class ChatMessage(models.Model):
    sender = models.CharField(max_length=10)  # 'user' or 'assistant'
    message_text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.message_text[:20]}..."