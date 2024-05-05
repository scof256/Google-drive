from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('create-tasks/', views.create_tasks, name='create_tasks'),
    path('execute-tasks/', views.execute_tasks, name='execute_tasks'),
    path('get-task-list/', views.get_task_list, name='get_task_list'),
    path('create-chat-message/', views.create_chat_message, name='create_chat_message'),
    path('get-chat-messages/', views.get_chat_messages, name='get_chat_messages'),
]