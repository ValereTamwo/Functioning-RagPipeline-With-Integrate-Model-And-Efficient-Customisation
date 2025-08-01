from typing import Dict, List
import uuid
from langchain.memory import ConversationBufferMemory

class SessionManager:
    def __init__(self):
        # Stockage en mémoire des sessions (suffisant pour un hackathon)
        self.sessions: Dict[str, ConversationBufferMemory] = {}
        self.responses: Dict[str, List[Dict]] = {}  

    def create_session(self) -> str:
        """Crée une nouvelle session et retourne son ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ConversationBufferMemory()
        self.responses[session_id] = []
        return session_id

    def add_response(self, session_id: str, subject: str, user_answer: int, correct_answer: int) -> None:
        """Ajoute une réponse à l'historique de la session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} n'existe pas.")
        self.responses[session_id].append({
            "subject": subject,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": user_answer == correct_answer
        })

    def get_session_history(self, session_id: str) -> ConversationBufferMemory:
        """Récupère l'historique de la session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} n'existe pas.")
        return self.sessions[session_id]

    def get_response_history(self, session_id: str) -> List[Dict]:
        """Récupère l'historique des réponses."""
        return self.responses.get(session_id, [])

    def end_session(self, session_id: str) -> None:
        """Termine une session et supprime ses données."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.responses[session_id]

    def get_difficulty_modifier(self, session_id: str) -> str:
        """Calcule un modificateur de difficulté basé sur les réponses précédentes."""
        responses = self.get_response_history(session_id)
        if not responses:
            return "normal"
        
        correct_count = sum(1 for r in responses if r["is_correct"])
        total = len(responses)
        success_rate = correct_count / total if total > 0 else 0

        if success_rate < 0.4:
            return "easy"  
        elif success_rate > 0.7:
            return "hard"  
        return "normal"