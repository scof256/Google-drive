from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("", views.index, name="index"),
    path("login/", views.Login, name="login"),
    path("signup/", views.SignUp, name="signup"),
    path("folder/<int:folderid>/", views.folder, name="folder"),
    path("addfolder/", views.addfolder, name="addfolder"),
    path("logout/", views.Logout, name="logout"),
    path('pdf/<int:file_id>/', views.view_pdf, name='view_pdf'),
    path('pdf_image/<int:file_id>/<int:page_num>/', views.view_pdf_page, name='view_pdf_page'),
    path('chat_with_document/<int:file_id>/', views.chat_with_document, name='chat_with_document'),
    path('all-files/', views.all_files, name='all_files'),
    path('all-files-guest/', views.all_files_guest, name='all_files_guest'),
    path('like-file/<int:file_id>/', views.like_file, name='like_file'),
    path('bookmark-file/<int:file_id>/<int:folder_id>/', views.bookmark_file, name='bookmark_file'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('chatbot2/', views.chatbot2, name='chatbot2'),
    path('summarize_the_document/<int:file_id>/', views.summarize_the_document, name='summarize_the_document'),
    path('summarize_next_page/<int:file_id>/', views.summarize_next_page, name='summarize_next_page'),
    path('clear_conversation_history/', views.clear_conversation_history, name='clear_conversation_history'),
    path('download-progress/<int:file_id>/', views.download_file, name='download_file'),
    path('research/', views.text_completion_view, name='research'),



]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns += staticfiles_urlpatterns()
