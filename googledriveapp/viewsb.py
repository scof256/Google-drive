from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from .models import Folder, File, YEAR_CHOICES, Bookmark
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from chromadb.utils import embedding_functions
from django.http import JsonResponse
from langchain.prompts import PromptTemplate
from django.http import StreamingHttpResponse
from django.http import HttpRequest
from django.utils.crypto import get_random_string
from django.urls import reverse


import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_openai.llms import OpenAI
from langchain_together.embeddings import TogetherEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from flashrank import Ranker
from langchain.chat_models import ChatOpenAI
from langchain import hub
from openai import OpenAI

from langchain.chains.question_answering import load_qa_chain
import markdown
from langchain_together import Together
from langchain.retrievers import SelfQueryRetriever



from groq import Groq
from langchain_groq import ChatGroq

import json
import logging

logger = logging.getLogger(__name__)

client1 = OpenAI(api_key= "2332ed28db876b7f1a700baa5a257464381df06b99e8213b4dade9a00091caa1", base_url="https://api.together.xyz/v1")
client = OpenAI(api_key="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", base_url="https://api.groq.com/openai/v1")

embedding = TogetherEmbeddings(together_api_key="2332ed28db876b7f1a700baa5a257464381df06b99e8213b4dade9a00091caa1", model="togethercomputer/m2-bert-80M-8k-retrieval")
embedding2 = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key="ERZcb0ybROkR3WZsp1qomQ2Qrwoy8KfqmH9iqXuF")
llm = ChatGroq(temperature=0, 
               groq_api_key="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", 
               model_name="llama3-70b-8192", 
               max_tokens=7800)
llm1= Together(
    model="openchat/openchat-3.5-1210",
    temperature=0,
    max_tokens=7000,
    top_k=0.2,
    together_api_key="2332ed28db876b7f1a700baa5a257464381df06b99e8213b4dade9a00091caa1"
)

def index(request):
    if request.user.is_authenticated:
        folders = Folder.objects.filter(folderuser=request.user)
        context = {'folders': folders}
        return render(request, 'googledriveapp/index.html', context)
    else:
        return redirect('login')

def all_files_guest(request):
    all_files = File.objects.all()
    context = {
        'all_files': all_files,
    }
    return render(request, 'googledriveapp/all_files_guest.html', context)

def all_files(request):
    if request.user.is_authenticated:
        all_files = File.objects.select_related('folder', 'folder__folderuser').all()
        context = {'all_files': all_files, 'user_folders': request.user.folder_set.all()}
        return render(request, 'googledriveapp/all_files.html', context)
    else:
        return redirect('login')

@login_required
def folder(request, folderid):
    if request.user.is_authenticated:
        folder_obj = get_object_or_404(Folder, id=folderid)
        files = File.objects.filter(folder=folder_obj)
        bookmarked_files = folder_obj.bookmarks.values_list('file', flat=True)

        if request.method == 'POST':
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                # Handle AJAX POST request for adding a file
                filetitle = request.POST.get('filetitle')
                year = request.POST.get('year')
                course = request.POST.get('course')
                course_unit = request.POST.get('course_unit')
                description = request.POST.get('description')
                file_obj = request.FILES.get('file')

                if filetitle and file_obj:
                    file_instance = File.objects.create(
                        filetitle=filetitle,
                        folder=folder_obj,
                        file=file_obj,
                        year=year,
                        course=course,
                        course_unit=course_unit,
                        description=description,
                        namespace=None  # Set namespace to None explicitly
                    )
                    # Preserve the original folderuser
                    file_instance.folder.folderuser = folder_obj.folderuser
                    file_instance.folder.save()

                    return JsonResponse({'success': True})
                else:
                    return JsonResponse({'success': False, 'error': 'Please provide both file and file title.'})
            else:
                # Handle regular form submission (not AJAX)
                filetitle = request.POST.get('filetitle')
                year = request.POST.get('year')
                course = request.POST.get('course')
                course_unit = request.POST.get('course_unit')
                description = request.POST.get('description')
                file_obj = request.FILES.get('file')

                if filetitle and file_obj:
                    file_instance = File.objects.create(
                        filetitle=filetitle,
                        folder=folder_obj,
                        file=file_obj,
                        year=year,
                        course=course,
                        course_unit=course_unit,
                        description=description,
                        namespace=None  # Set namespace to None explicitly
                    )
                    # Preserve the original folderuser
                    file_instance.folder.folderuser = folder_obj.folderuser
                    file_instance.folder.save()

        context = {
            'folder': folder_obj,
            'files': files,
            'year_choices': YEAR_CHOICES,
            'bookmarked_files': bookmarked_files,
        }
        return render(request, 'googledriveapp/folder.html', context)
    else:
        return redirect('login')
        
@login_required
def like_file(request, file_id):
    file_obj = get_object_or_404(File, id=file_id)
    if request.user in file_obj.likes.all():
        file_obj.likes.remove(request.user)
        liked = False
    else:
        file_obj.likes.add(request.user)
        liked = True
    data = {
        'liked': liked,
        'likes_count': file_obj.likes.count()
    }
    return JsonResponse(data)

@login_required
def bookmark_file(request, file_id, folder_id):
    file_obj = get_object_or_404(File, id=file_id)
    folder_obj = get_object_or_404(Folder, id=folder_id, folderuser=request.user)

    bookmark, created = Bookmark.objects.get_or_create(
        file=file_obj,
        folder=folder_obj,
        user=request.user
    )

    if not created:
        bookmark.delete()

    return redirect('all_files')

@login_required
def addfolder(request):
    if request.method == 'POST':
        if request.user.is_authenticated:
            folder_name = request.POST.get('foldername')
            folder_desc = request.POST.get('desc')
            if folder_name and folder_desc:
                try:
                    folder = Folder.objects.create(foldername=folder_name, folderdesc=folder_desc, folderuser=request.user)
                    return redirect('index')
                except Exception as e:
                    messages.error(request, str(e))
            else:
                messages.error(request, 'Please provide both folder name and description.')
        else:
            messages.error(request, 'You must be logged in to create a folder.')
    return redirect('index')

def SignUp(request):
    if request.user.is_authenticated:
        return redirect('index')
    else:
        if request.method == 'POST':
            username = request.POST['username']
            email = request.POST['email']
            password = request.POST['password']
            cpassword = request.POST['cpassword']
            firstname = request.POST['fname']
            lname = request.POST['lname']
            if username and password and email and cpassword and firstname and lname:
                if password == cpassword:
                    user = User.objects.create_user(username, email, password)
                    user.first_name = firstname
                    user.last_name = lname
                    user.save()
                    if user:
                        messages.success(request, "User Account Created")
                        return redirect("login")
                    else:
                        messages.error(request, "User Account Not Created")
                else:
                    messages.error(request, "Password Not Matched")
                    return redirect("signup")
        return render(request, 'googledriveapp/signup.html')

def Login(request):
    if request.user.is_authenticated:
        return redirect("index")
    else:
        if request.method == 'POST':
            username = request.POST['username']
            password = request.POST['password']
            if username and password:
                user = authenticate(username=username, password=password)
                if user is not None:
                    login(request, user)
                    return redirect('index')
        return render(request, 'googledriveapp/login.html')

def Logout(request):
    logout(request)
    return redirect("index")

from django.http import HttpResponse
from .models import File
from PIL import Image
import fitz
import io

def view_pdf(request, file_id):
    file_obj = get_object_or_404(File, id=file_id)

    doc = fitz.open(file_obj.file.path)

    # Get the total number of pages
    total_pages = doc.page_count
    print(total_pages)

    # Get the URL of the first page image
    first_page_url = reverse('view_pdf_page', args=[file_id, 1])

    context = {
        'file_obj': file_obj,
        'total_pages': total_pages,
        'first_page_url': first_page_url,
    }

    return render(request, 'googledriveapp/chat.html', context)


def summarize_the_document(request, file_id):
    file_obj = get_object_or_404(File, id=file_id)
    
    if request.method == 'POST':
        current_page = int(request.POST.get('page_number', 1))
        summary = summarize_page(file_obj, current_page)
        summary_markdown = markdown.markdown(summary)
        
        # Open the PDF file
        doc = fitz.open(file_obj.file.path)
        
        # Get the total number of pages
        total_pages = doc.page_count
        
        return JsonResponse({'summary': summary_markdown, 'total_pages': total_pages})
    else:
        context = {
            'pdf_url': request.build_absolute_uri(file_obj.file.url),
            'file_name': file_obj.filetitle,
            'file_obj': file_obj,
            'page_num': 1,  # Pass the initial page number to the template
        }
        return render(request, 'googledriveapp/summarize.html', context)
    
def view_pdf_page(request, file_id, page_num):
    file_obj = get_object_or_404(File, id=file_id)

    # Open the PDF file
    doc = fitz.open(file_obj.file.path)

    # Get the total number of pages
    total_pages = doc.page_count

    # Get the specified page of the PDF
    page = doc[page_num - 1]

    # Render the page as an image
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))

    # Convert the pixmap to a PIL image
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Convert the PIL image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    # Create an HTTP response with the image data
    response = HttpResponse(content_type='image/png')
    response.write(image_bytes)

    # Add the total page count to the response headers
    response['Total-Pages'] = total_pages

    return response

"""
def chat_with_document(request, file_id):
    file_obj = get_object_or_404(File, id=file_id)
    namespace = file_obj.namespace

    if request.method == 'POST':
        question = request.POST.get('question')

        # Load the embeddings from the database
        vectorstore = Chroma(collection_name=namespace, embedding_function=embedding2, persist_directory="googledriveapp/persist")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Load the question-answering chain
        #chain = load_qa_chain(llm=llm, chain_type="stuff")
        
        "ms-marco-MiniLM-L-12-v2"
        "ms-marco-TinyBERT-L-2-v2"
        compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
                )
        
        chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                  retriever=compression_retriever,
                                                  return_source_documents=True)
        # Find the relevant documents based on the question
        
        compressed_docs = compression_retriever.get_relevant_documents (question) 
        chat_history=[compressed_docs]      
        # Get the answer from the relevant documents
        answer = chain({"question": question, 'chat_history': compressed_docs})
        print(answer)

        # Extract the page numbers from the relevant documents
        page_numbers = [doc.metadata['page']+1 for doc in compressed_docs]
        source = [doc.metadata['source'] for doc in compressed_docs]

        context = {
            'answer': answer["output_text"],
            'source': source,
            'page_numbers': page_numbers,
        }
        return JsonResponse(context)
    else:
        context = {
            'pdf_url': request.build_absolute_uri(file_obj.file.url),
            'file_name': file_obj.filetitle,
            'file_obj': file_obj,
        }
        return render(request, 'googledriveapp/chat.html', context)
"""
@login_required
def chat_with_document(request, file_id):
    file_obj = get_object_or_404(File, id=file_id)
    if not file_obj.namespace:
        loader = PyPDFLoader(file_obj.file.path)
        documents = loader.load()
        if documents:
            text_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len)
            texts = text_splitter.split_documents(documents)
            if texts:
                namespace = f"file_{file_obj.id}_{get_random_string(length=8)}"
                file_obj.namespace = namespace
                file_obj.save()
                client = Chroma.from_documents(texts, embedding2, collection_name=namespace, persist_directory="googledriveapp/persist")
    namespace = file_obj.namespace
    if request.method == 'POST':
        question = request.POST.get('question')
        vectorstore = Chroma(collection_name=namespace, embedding_function=embedding2, persist_directory="googledriveapp/persist")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=6)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        compressed_docs = compression_retriever.get_relevant_documents(question)
        prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use 10 sentences maximum. Keep the answer as concise as possible. Always say 
        "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        #QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", 
                                         retriever=compression_retriever, 
                                         return_source_documents=True, 
                                         chain_type_kwargs={"prompt": prompt})
        result = qa({"query": question})
        answer = result['result']
        relevant_docs = compression_retriever.get_relevant_documents(question)
        page_numbers = [doc.metadata['page'] + 1 for doc in relevant_docs]
        source_docs = [doc for doc in relevant_docs]
        source_and_page = []
        for doc, page in zip(source_docs, page_numbers):
            source_and_page.append({
                'source': doc.page_content,  # or doc.content, depending on your Document object
                'page': page
            })
        context = {
            'pdf_url': request.build_absolute_uri(file_obj.file.url),
            'file_name': file_obj.filetitle,
            'question': question,
            'answer': answer,
            'page_numbers':page_numbers,
            'source_and_page': source_and_page
        }
        return JsonResponse(context) 
    else:
        context = {
            'pdf_url': request.build_absolute_uri(file_obj.file.url),
            'file_name': file_obj.filetitle,
            'file_obj': file_obj,
        }
    return render(request, 'googledriveapp/chat.html', context)  


#summarise the document

import markdown

def summarize_page(file_obj, page_number):
    file_obj.current_page = page_number
    file_obj.save()
    page_number=page_number-1
    namespace = file_obj.namespace
    client = Chroma(collection_name=namespace, embedding_function=embedding2, persist_directory="googledriveapp/persist")
    vectorstore = client.as_retriever(search_type="similarity", search_kwargs={'filter': {'page':page_number},"k": 5})

    template = """Use the following pieces of context to summarize the content of the current page. 
    Keep the summary as concise as possible and easy to read. 
    summarize each paragragh. format your text properly and use bullet points format to summarize paragraphs. 
    only summarize whats on the page. include breif of statistics if given.  if there is nothing or just image dont makeup your own stuff..
    dont give your own introductions or conclusion, only summaries of context given. dont say things like Here is a concise summary of the current page in bullet points.
    just give the summaries. format your text properly. please dont give your own options or answer questions in the context.
    {context} :
    here is the {question}
    
    """
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    result = qa({"query": "only summarize the context given. dont answer any quesions in the context"})
    summary = result['result']
    
    print(summary)
    return summary


from django.http import HttpResponse, JsonResponse
import markdown



def summarize_next_page(request, file_id):
    file_obj = get_object_or_404(File, id=file_id)
    
    if request.method == 'POST':
        current_page = int(request.POST.get('page_number', 1))
        summary = summarize_page(file_obj, current_page)
        summary_markdown = markdown.markdown(summary)
        messages = [
            {
                'sender': 'bot',
                'text': summary_markdown,
                'page_numbers': []
            }
        ]
        return JsonResponse({'message': 'POST request received', 'summary': summary_markdown})
    else:
        summary = summarize_page(file_obj, 1)
        summary_markdown = markdown.markdown(summary)
        messages = [
            {
                'sender': 'bot',
                'text': summary_markdown,
                'page_numbers': []
            }
        ]
        context = {
            'pdf_url': request.build_absolute_uri(file_obj.file.url),
            'file_name': file_obj.filetitle,
            'file_obj': file_obj,
            'messages': messages,
            'page_num': 1  # Pass the initial page number to the template
        }
        return render(request, 'googledriveapp/summarize.html', context)
    
import markdown
from django.http import JsonResponse, StreamingHttpResponse


from django.contrib.sessions.models import Session

from django.contrib.sessions.backends.db import SessionStore


@csrf_exempt
def chatbot(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question = data.get("question")
            logger.info(f"Received question: {question}")
            if question:
                try:
                    # Get the conversation history from the session
                    session = SessionStore(session_key=request.session.session_key)
                    conversation_history = session.get('conversation_history', [])

                    # Add the new question to the conversation history
                    conversation_history.append({"role": "user", "content": question})

                    response_generator = generate_response(conversation_history)

                    def stream_response():
                        for chunk in response_generator:
                            yield chunk

                    # Update the session with the new conversation history
                    session['conversation_history'] = conversation_history
                    session.save()

                    return StreamingHttpResponse(stream_response(), content_type='text/plain')
                except Exception as e:
                    logger.exception("Error generating response")
                    return JsonResponse({"error": str(e)}, status=500)
            else:
                logger.warning("Empty question received")
                return JsonResponse({"error": "Empty question"}, status=400)
        except json.JSONDecodeError as e:
            logger.exception("Invalid JSON data")
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
    return render(request, "googledriveapp/chatbot.html")

def generate_response(conversation_history):
    # Use OpenAI's chat model to generate a response with the conversation history
    chat_completion = client.chat.completions.create(
        messages=conversation_history + [{"role": "system", "content": """ These are your commands, dont introduce yourself. only reply to what the user has asked. be friendly. dont talk out topics the user has not asked. properly format your text. when you have answered what the user wanted stop. when replying to the users topic, Discuss, describe and fully elaborate each point with examples as if you were a human Ugandan University Professor helping university students pass exams. Be professional and factual. Answer based on context of question asked. Use simple english. Give more than 10 points. Fully explain and elaborate each point with an example. Where possible give references or quotations. Examples should be relevant to uganda. Use essays with introduction, body and conclusion. Rewrite all content and answers like a human being. rephrase all content. Change the syntax and vocabulary. Recheck and verify your answers to eliminate errors, and give the most correct answer. """}],
        stream=True,
        model="llama3-70b-8192"
    )

    for chunk in chat_completion:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                # Update the conversation history with the assistant's response
                conversation_history.append({"role": "assistant", "content": delta.content})
                yield delta.content
    # Store the updated conversation history in the session
    request.session['conversation_history'] = conversation_history
    request.session.modified = True
    
    
@csrf_exempt
def clear_conversation_history(request):
    if request.method == 'POST':
        try:
            session = SessionStore(session_key=request.session.session_key)
            if 'conversation_history' in session:
                del session['conversation_history']
            session.save()
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
import json

import markdown

@login_required
@csrf_exempt
def chatbot2(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question = data.get("question")
            logger.info(f"Received question: {question}")
            if question:
                try:
                    # Get the conversation history from the session
                    session = SessionStore(session_key=request.session.session_key)
                    conversation_history = session.get('conversation_history', [])

                    # Add the new question to the conversation history
                    conversation_history.append({"role": "user", "content": question})

                    response_generator = generate_response(conversation_history)

                    def stream_response():
                        for chunk in response_generator:
                            yield chunk

                    # Update the session with the new conversation history
                    session['conversation_history'] = conversation_history
                    session.save()

                    return StreamingHttpResponse(stream_response(), content_type='text/plain')
                except Exception as e:
                    logger.exception("Error generating response")
                    return JsonResponse({"error": str(e)}, status=500)
            else:
                logger.warning("Empty question received")
                return JsonResponse({"error": "Empty question"}, status=400)
        except json.JSONDecodeError as e:
            logger.exception("Invalid JSON data")
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
    return render(request, "googledriveapp/chatbot.html")

def generate_response2(conversation_history):
    # Use OpenAI's chat model to generate a response with the conversation history
    chat_completion = client.chat.completions.create(
        messages=conversation_history + [{"role": "system", "content": """ These are your commands, dont introduce yourself. only reply to what the user has asked. be friendly. dont talk out topics the user has not asked. properly format your text. when you have answered what the user wanted stop. when replying to the users topic, Discuss, describe and fully elaborate each point with examples as if you were a human Ugandan University Professor helping university students pass exams. Be professional and factual. Answer based on context of question asked. Use simple english. Give more than 10 points. Fully explain and elaborate each point with an example. Where possible give references or quotations. Examples should be relevant to uganda. Use essays with introduction, body and conclusion. Rewrite all content and answers like a human being. rephrase all content. Change the syntax and vocabulary. Recheck and verify your answers to eliminate errors, and give the most correct answer. """}],
        stream=True,
        model="llama3-70b-8192"
    )

    for chunk in chat_completion:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                # Update the conversation history with the assistant's response
                conversation_history.append({"role": "assistant", "content": delta.content})
                yield delta.content
    # Store the updated conversation history in the session
    request.session['conversation_history'] = conversation_history
    request.session.modified = True
    
import os
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings


def get_file_path(file_id):
    try:
        file_obj = File.objects.get(id=file_id)
        file_path = os.path.join(settings.MEDIA_ROOT, file_obj.file.name)
        return file_path
    except File.DoesNotExist:
        return None

def download_file(request, file_id):
    # Get the file path based on the file ID
    file_path = get_file_path(file_id)

    # Open the file and get its size
    file_size = os.path.getsize(file_path)

    # Create a generator to yield file chunks
    def file_iterator(file_path, chunk_size=1024):
        with open(file_path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    # Create a streaming response
    response = StreamingHttpResponse(file_iterator(file_path), content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
    response['Content-Length'] = file_size

    # Create a generator to yield progress data
    def progress_generator(response, file_size):
        total_bytes_sent = 0
        for chunk in response.streaming_content:
            yield chunk
            total_bytes_sent += len(chunk)
            progress = int(total_bytes_sent / file_size * 100)
            yield f'data: {progress}\n\n'

    # Stream the response with progress data
    return StreamingHttpResponse(progress_generator(response, file_size), content_type='text/event-stream')

from django.contrib.sessions.models import Session
import json

client = OpenAI(api_key="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", 
                base_url="https://api.groq.com/openai/v1")

def text_completion_tool(query_space: str, documents):
    sys_prompt= f"""You are an expert academic report writer. Use the following information only without removing any 
                facts and write in apa style with parenterical citations and references using on the information given below.\n###INFORMATION:{documents}.\n###REPORT:"""
    
    messages = [
        {
            "role": "system", "content": sys_prompt,
            "role": "user", "content":   "follow the rules below to write a well explained apa format content"
                                        "Please provide content based solely on the provided sources. "
                                        "When referencing information from a source, "
                                        "cite the appropriate source(s) using parenterical citation. "
                                        "Every answer should include at least one source citation. "
                                        "Only cite a source when you are explicitly referencing it. "
                                        "provide the user with a link to the pdf file in the references section after every reference"
                                        "make sure you use the correct and right references provided on the right content and links"
                                        "format the text, headings, paragraphs and content in the proper way"
                                        "dont create information or facts that don't exit from the context. only use informatation or facts given"
                                        "If none of the sources are helpful, you should indicate that. "
                                        "if you have recieved empty context don't reply"
                                        "refine the existing answer. "
                                        "If the provided sources are not helpful, you will repeat the existing answer."
                                        "\nBegin refining!"
                                        "\n------\n"
                                        f"use only this infrormation{documents} as reference to write"
                                        "\n------\n"
                                        f"write a well explained apa format content with the title {query_space}\n"
                                        "Answer: "
                                        "References with correct links: ",
        }
    ]
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0,
        max_tokens=7800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content.strip()


from django.http import HttpResponse
from openai import OpenAI
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.readers.semanticscholar import SemanticScholarReader
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
from llama_index.core import PromptTemplate

def text_completion_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query_space = data.get("question")
        if query_space:
            generative_model = "llama3-70b-8192"
            Settings.llm = Groq(temperature=0, model=generative_model, api_key="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", api_base="https://api.groq.com/openai/v1")
            Settings.embed_model = TogetherEmbedding(model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key="2332ed28db876b7f1a700baa5a257464381df06b99e8213b4dade9a00091caa1", api_base="https://api.together.xyz/v1")
            s2reader = SemanticScholarReader(api_key="ftv3M685Tb2DZ0s2NDhD83EW4UIGUWZc5gQyApC8")
            documents = s2reader.load_data(query=query_space, limit=10, full_text=False)
            print(documents)
            response = text_completion_tool(query_space, documents)
            formatted_answer = markdown.markdown(response)
            return HttpResponse(formatted_answer)
        else:
            return JsonResponse({"error": "Please provide a valid query."})
    return render(request, "googledriveapp/researchbot.html")