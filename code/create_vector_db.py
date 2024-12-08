import os
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS


# def load_pdf_documents(path):
#     """Load a PDF file and extract its content."""
#     loader = PyPDFLoader(path)
#     documents = loader.load()
#     print(documents)
#     return documents


def load_documents(path):
    """Load .txt files and extract their content."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path '{path}' does not exist.")

    docs = []
    for file_name in os.listdir(path):
        loader = TextLoader(os.path.join(path, file_name), encoding='UTF-8')
        doc = loader.load()
        docs.extend(doc)
    print(f'Loaded {len(docs)} documents.')
    return docs


def generate_embeddings():
    """Initialize the embedding model."""
    # embedding_model =  SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_model =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model


def create_vector_db(documents, file_path):
    """Create a FAISS vector database from documents."""
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    vec_db = FAISS.from_documents(documents=docs, embedding=generate_embeddings())
    # Save vec_db
    vec_db.save_local(file_path)
    return vec_db


def main():
    docs = load_documents(path="D:\\Learn\\LLMs\\code\\AI medical assistant\\documents to learn from")
    vec_db = create_vector_db(documents=docs, file_path="D:\\Learn\\LLMs\\code\\AI medical assistant\\vec_db")


if __name__ == "__main__":
    main()
