from firebase_init import db
from firebase_admin import firestore
import json
import time # Vom folosi time pentru a simula ID-uri unice

HOTEL_CONTEXT = {} # Variabila globală pentru stocarea datelor cheie

def load_hotel_context():
    """Încarcă datele din colecția 'hotel' pentru a fi folosite de agentul AI."""
    
    global HOTEL_CONTEXT
    if db is None:
        print("Nu se poate citi, conexiunea Firebase a eșuat la inițializare.")
        return {}

    try:
        print("\nÎncep citirea colecției 'hotel'...")
        
        hotel_refs = db.collection('hotels').stream() 
        doc_count = 0
        
        for doc in hotel_refs:
            # 1. AFISARE LA NIVEL DE DOCUMENT (Afișează fiecare document pe măsură ce este citit)
            print(f"  > Citit documentul ID: {doc.id}")
            # Opțional: Afișează conținutul documentului (poate fi mult)
            # print(f"    Detalii: {doc.to_dict()}") 
            
            HOTEL_CONTEXT[doc.id] = doc.to_dict()
            doc_count += 1

        if doc_count > 0:
            print(f"\n✅ Citire finalizată. Total documente citite: {doc_count}.")
        else:
            print("\n⚠️ Avertisment: Colecția 'hotel' a fost găsită, dar nu conține niciun document.")
            
        return HOTEL_CONTEXT
        
    except Exception as e:
        print(f"❌ Eroare la citirea datelor hotelului: {e}")
        return {}

def log_test_activity(agent_name="Test Agent", test_message="Conexiune INSERT reușită"):
    """
    Înregistrează o activitate de test în colecția 'agent_logs'.
    
    :return: True dacă inserarea a reușit, False altfel.
    """
    if db is None:
        print("Nu se poate scrie, clientul Firestore nu este inițializat.")
        return False
        
    try:
        # 1. Creează referința către colecția de logs
        log_ref = db.collection('hotels')
        
        test_log_data = {
            'agent': agent_name,
            'log_type': 'SYSTEM_TEST',
            'message': test_message,
            'timestamp': firestore.SERVER_TIMESTAMP, # Timestamp-ul serverului
            'test_id': f"TEST-{int(time.time())}" # Un ID unic bazat pe timp
        }
        
        # 2. Adaugă documentul. Firestore generează automat un ID.
        update_time, doc_ref = log_ref.add(test_log_data)
        
        print(f"\n✅ Inserare reușită! Document adăugat în: {doc_ref.path}")
        print(f"   La ora (locală): {update_time}")
        return True
        
    except Exception as e:
        print(f"❌ Eroare la scrierea (INSERT) în Firestore: {e}")
        return False


# --- Punctul de Intrare (Main) - Modificat ---

if __name__ == "__main__":
    
    # 1. Testează citirea datelor (opțional, dar bun)
    context_data = load_hotel_context()
    
    if context_data:
        print("\n✅ Datele hotelului încărcate în memorie (Cache):")
        # 2. AFISARE LA FINAL (Afișează tot ce a fost salvat în memorie)
        print(json.dumps(context_data, indent=4))
    else:
        print("\n⚠️ Nu s-au putut încărca date din Firestore.")
        
    # 2. Testează inserarea datelor
    #log_test_activity() 
    
    # 3. Poți verifica manual Firestore Console (colecția 'agent_logs')