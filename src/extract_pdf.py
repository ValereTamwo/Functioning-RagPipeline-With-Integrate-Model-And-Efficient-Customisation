import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_and_split_pdf(pdf_path: str) -> list[str]:
    """
    Extrait le texte d’un PDF et le divise en chunks.
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF.
    
    Returns:
        list[str]: Liste des chunks de texte.
    """
    # Extraire le texte
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Diviser en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    with open("chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")
    
    return chunks