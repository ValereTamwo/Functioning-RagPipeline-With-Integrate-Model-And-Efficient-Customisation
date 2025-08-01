from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import os

def create_rag_pipeline(vectorstore: Chroma) -> RetrievalQA:
    """
    Crée un pipeline RAG avec Gemini et la base vectorielle.
    """
    # Configurer l’API de Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY") or "",
        temperature=0.7
    )
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Tu es un assistant expert qui génère des quiz au format strict, sans aucun texte supplémentaire.
    Ta tâche est de créer un scénario éducatif sur le code électoral camerounais en te basant sur un sujet et un contexte fournis.
    Tu dois répondre UNIQUEMENT avec le format demandé. N'ajoute aucune introduction, salutation ou conclusion.

    --- EXEMPLE ---
    **Sujet** : Conditions pour être électeur.
    **Contexte** : Article L. 25 : Sont électeurs, les Camerounais des deux sexes, âgés de vingt (20) ans révolus, jouissant de leurs droits civiques et politiques, et non frappés par l’une des incapacités prévues par la loi.
    
    **Réponse Attendue** :
    - Scénario : Amina vient de fêter son 20ème anniversaire et est très excitée à l'idée de voter pour la première fois aux prochaines élections municipales. Elle se rend à l'antenne ELECAM de son quartier pour s'inscrire sur les listes électorales. L'agent au guichet lui demande de présenter ses documents pour vérifier si elle remplit toutes les conditions requises par la loi. Amina présente sa carte d'identité nationale. L'agent doit vérifier plusieurs critères avant de valider son inscription.
    - Choix :
      1. Être de nationalité camerounaise et avoir au moins 18 ans.
      2. Être de nationalité camerounaise, avoir au moins 20 ans et jouir de ses droits civiques.
      3. Être résident au Cameroun depuis 5 ans et avoir au moins 20 ans.
    - Réponse correcte : 2
    - Explication : Selon l'article L. 25 du Code Électoral, il faut être camerounais, avoir 20 ans révolus et jouir de ses droits civiques et politiques pour être électeur.
    --- FIN DE L'EXEMPLE ---

    --- TÂCHE À RÉALISER ---
    **Sujet** : {question}
    **Contexte** : {context}

    **Ta Réponse** :
    """
    )
    
    # Configurer le pipeline RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",    
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
        
    )   
    
    return qa_chain
 
