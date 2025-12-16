import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Încarcă variabilele din fișierul .env (asigură-te că PATH-ul e corect)
load_dotenv() 

def initialize_firebase_app():
    """Inițializează conexiunea la Firebase și returnează clientul Firestore."""
    
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    
    if not cred_path or not os.path.exists(cred_path):
        print("❌ Eroare: Calea către fișierul de credențiale nu este validă. Verifică .env.")
        return None
        
    try:
        # Folosește credențialele Service Account (cheia ta JSON)
        cred = credentials.Certificate(cred_path)
        
        # Inițializează aplicația Firebase
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            print("✅ Conexiune Firebase stabilită cu succes.")
        
        # Returnează clientul Firestore pentru a interacționa cu baza de date
        return firestore.client()
        
    except Exception as e:
        print(f"❌ Eroare la inițializarea Firebase: {e}")
        return None

db = initialize_firebase_app()
