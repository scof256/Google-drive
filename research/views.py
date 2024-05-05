from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from .ai_assistant import task_creation_agent, execute_task
from .forms import ObjectiveForm
import json

def index(request):
    form = ObjectiveForm()
    return render(request, 'index.html', {'form': form})

def create_tasks(request):
    if request.method == 'POST':
        form = ObjectiveForm(request.POST)
        if form.is_valid():
            objective = form.cleaned_data['objective']
            task_list = task_creation_agent(objective)
            return JsonResponse({'task_list': task_list})
    return JsonResponse({'error': 'Invalid request method'})

def execute_tasks(request):
    if request.method == 'POST':
        task_list = json.loads(request.POST.get('task_list', '[]'))
        objective = request.POST.get('objective', '')
        for task in task_list:
            if task["status"] == "incomplete":
                execute_task(task, task_list, objective)
        return JsonResponse({'task_list': task_list})
    return JsonResponse({'error': 'Invalid request method'})

def get_task_list(request):
    if request.method == 'GET':
        task_list = request.session.get('task_list', [])
        return JsonResponse({'task_list': task_list})
    return JsonResponse({'error': 'Invalid request method'})


# views.py
from django.http import JsonResponse
from .models import ChatMessage

def create_chat_message(request):
    if request.method == 'POST':
        sender = request.POST.get('sender')
        message_text = request.POST.get('message_text')
        chat_message = ChatMessage.objects.create(sender=sender, message_text=message_text)
        return JsonResponse({'message': 'Chat message created successfully'})
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def get_chat_messages(request):
    chat_messages = list(ChatMessage.objects.values('sender', 'message_text', 'timestamp'))
    return JsonResponse({'chat_messages': chat_messages})