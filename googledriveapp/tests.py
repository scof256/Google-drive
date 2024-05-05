from django.test import TestCase

# Create your tests here.
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.vector_stores.types import ExactMatchFilter
from llama_index.llms import together
from llama_index.core.prompts import PromptTemplate

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,ServiceContext
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
import os

def summarize_page_llama(index, page_number):
    """Summarizes a specific page from a LlamaIndex VectorStore.

    Args:
        index (VectorStoreIndex): The LlamaIndex VectorStore to query.
        page_number (int): The page number to summarize.

    Returns:
        str: The summarized content of the specified page.
    """

    # Define filters for page number
    filters = ExactMatchFilter(page_number=page_number)
    
    # Define prompt template
    template = """
    ## Page {page_number} Summary:

    Based on the context below, provide a concise summary of the page content, preferably using bullet points.

    **Context:**

    {context}
    """
    prompt_template = PromptTemplate(input_variables=["page_number", "context"], template=template)

    # Set up service context with Llama LLM
    llm = llm
    service_context = ServiceContext.from_defaults(llm=llm)

    # Query the index and retrieve relevant context
    response = index.query(f"Summarize page {page_number}", filters=filters, service_context=service_context)
    context = response.response

    # Format prompt with retrieved context
    prompt = prompt_template.format(page_number=page_number, context=context)

    # Generate summary using the LLM
    summary = llm(prompt)
    return summary.strip()

# Example Usage (assuming you already have a LlamaIndex index)
index = construct_index("path/to/your/data")  # Replace with your index construction

page_number = 5  # Replace with desired page number
summary = summarize_page_llama(index, page_number)
print(summary)

client = Chroma(collection_name=namespace, embedding_function=embedding2, persist_directory="googledriveapp/persist")
