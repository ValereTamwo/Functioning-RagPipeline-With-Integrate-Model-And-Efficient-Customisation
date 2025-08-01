from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks: list[str], persist_directory: str = "./chroma_db") -> Chroma:
    """
    Crée une base de données vectorielle à partir des chunks de texte.
    
    Args:
        chunks (list[str]): Liste des chunks de texte.
        persist_directory (str): Dossier pour sauvegarder la base.
    
    Returns:
        Chroma: Instance de la base vectorielle.
    """
    # Configurer le modèle d’embedding
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Créer et persister la base vectorielle
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

def load_vector_store(persist_directory: str = "./chroma_db") -> Chroma:
    """
    Charge une base de données vectorielle existante.
    
    Args:
        persist_directory (str): Dossier contenant la base.
    
    Returns:
        Chroma: Instance de la base vectorielle.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)