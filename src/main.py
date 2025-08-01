from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from extract_pdf import extract_and_split_pdf
from vector_store import create_vector_store, load_vector_store
from rag_pipeline import create_rag_pipeline
import os
import re
import uuid
import base64
import requests
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  
from fastapi.staticfiles import StaticFiles
import logging
import os
from datetime import datetime

# Configuration du logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  
log_file = os.path.join(LOG_DIR, f"quiz_parser_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file), 
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)


load_dotenv()

GOOGLE_API_KEY = ""

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)



STATIC_FILES_DIR = "static_files"
IMAGES_DIR = os.path.join(STATIC_FILES_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True) 


app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

class QueryInput(BaseModel):
    query: str

qa_chain = None

@app.on_event("startup")
async def startup_event():
    global qa_chain
    if not os.path.exists("./chroma_db"):
        print("Création de la base de données vectorielle...")
        chunks = extract_and_split_pdf("/home/franck/BrightCitizenRoad/Backend/data/Cameroon-Electoral-Code-French.pdf")
        vectorstore = create_vector_store(chunks)
    else:
        print("Chargement de la base de données vectorielle...")
        vectorstore = load_vector_store()
    qa_chain = create_rag_pipeline(vectorstore)
    print("Pipeline RAG prêt.")

def parse_quiz_from_string(raw_text: str):
    try:
        pattern = re.compile(r"- Scénario : (.*?)\n\s*- Choix :\n\s*1\. (.*?)\n\s*2\. (.*?)\n\s*3\. (.*?)\n\s*- Réponse correcte : (\d+)\s*\n- Explication : (.*)", re.DOTALL)
        match = pattern.search(raw_text.strip())
        if not match: return None
        scenario, choice1, choice2, choice3, correct_answer_id, explanation = match.groups()
        return {"scenario": scenario.strip(), "choices": [{"id": 1, "text": choice1.strip()}, {"id": 2, "text": choice2.strip()}, {"id": 3, "text": choice3.strip()}], "correct_answer_id": int(correct_answer_id), "explanation": explanation.strip()}
    except Exception:
        return None

def generate_and_save_image_locally(scenario_text: str) -> dict | None:
    """
    Génère une image, la sauvegarde localement et retourne son chemin absolu et son URL relative.
    """
    if not GOOGLE_API_KEY:
        print("Variable GOOGLE_API_KEY manquante. Génération d'image ignorée.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-generate-content?key={GOOGLE_API_KEY}"
    image_prompt = (f"An illustrative scene for this scenario; "
                    f"La scène se déroule au Cameroun. Évite d'afficher du texte. Scène : {scenario_text}")
    payload = {"contents": [{"parts": [{"text": image_prompt}]}], "generationConfig": {"responseMimeType": "image/png"}}

    try:
        print("Envoi de la requête de génération d'image...")
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        base64_image_data = response_data['contents'][0]['parts'][0]['inlineData']['data']
        image_bytes = base64.b64decode(base64_image_data)

        file_name = f"{uuid.uuid4()}.png"
        absolute_file_path = os.path.abspath(os.path.join(IMAGES_DIR, file_name))

        print(f"Sauvegarde de l'image sur le chemin local : {absolute_file_path}")
        with open(absolute_file_path, "wb") as f: 
            f.write(image_bytes)
        
      
        url_path = f"/static/images/{file_name}"

        return {
            "absolute_path": absolute_file_path,
            "url_path": url_path
        }

    except Exception as e:
        print(f"Une erreur est survenue lors de la génération/sauvegarde de l'image : {e}")
        return None


@app.post("/generate-quiz")
async def generate_quiz(query: QueryInput):
    result = qa_chain.invoke({"query": query.query})
    raw_llm_output = result.get("result")
    parsed_quiz = parse_quiz_from_string(raw_llm_output)

    if not parsed_quiz:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"error": "Impossible d'analyser la réponse texte du modèle.", "raw_response_from_ai": raw_llm_output})
    if parsed_quiz:
        logger.info(f"Parsed Quiz: {parsed_quiz}")
    else:
        logger.error(f"Failed to parse quiz for query: {query.query}")
    #image_info = generate_and_save_image_locally(parsed_quiz["scenario"])
    image_info =None
    if image_info:
        parsed_quiz["image_absolute_path"] = image_info["absolute_path"]
        parsed_quiz["image_url"] = image_info["url_path"]
    else:
        parsed_quiz["image_absolute_path"] = None
        parsed_quiz["image_url"] = None

    return {
        "quiz": parsed_quiz,
        "source_documents": [doc.page_content for doc in result["source_documents"]]
    }


@app.get("/")
async def health():
    return {"message": "api is running"}
