from langchain_chroma import Chroma
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

###chroma持久化
#embeddings函数

model_path = './bge-m3'
from chromadb.utils import embedding_functions

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_path)

#客户端
client = chromadb.PersistentClient(path="./chromadb")
collection = client.create_collection(name="Building_Design_Codes", embedding_function=sentence_transformer_ef)

# document
file_path = (
    "./data/建筑设计常用规范速查手册第四版.pdf"
)
loader = PDFMinerLoader(file_path,extract_images=True)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)
# 添加文档
for doc in splits:
    collection.add(
        ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
    )
