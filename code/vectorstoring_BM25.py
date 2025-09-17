import os
import json
import time

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from kiwipiepy import Kiwi

with open("../data/documents.jsonl", "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

print(type(docs))
print(docs[0])

# embeddings
embedding=HuggingFaceEmbeddings(model_name='nlpai-lab/KURE-v1')

# tokenizer
kiwi_tokenizer = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi_tokenizer.tokenize(text)]


# time
start_time = time.time()

# text_splitter
text_splitter = SemanticChunker(embeddings=embedding)

# text (Langchain Document 형식으로 변환)
lc_docs = [
    Document(page_content=d["content"], metadata={"docid": d["docid"]})
    for d in docs
]

# text split to chunks
chunks = text_splitter.split_documents(lc_docs)

# text , metadatas 분리
texts = [c.page_content for c in chunks]
metadatas = [c.metadata for c in chunks]

# metadatas = [{"docid": d.get("docid"), "src": d.get("src")} for d in docs] # ids = [d["docid"] for d in docs] 

vectorstore = FAISS.from_texts(texts = texts, embedding=embedding, metadatas=metadatas,) #ids=ids)
vectorstore.save_local("faiss_index_KURE-v1_semantic") 
# vectorstore = FAISS.load_local("faiss_index", HuggingFaceEmbeddings, allow_dangerous_deserialization=True)

end_time = time.time()

print(vectorstore)
print(end_time - start_time)
# embeddings = get_embeddings_in_batches(docs)

