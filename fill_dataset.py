from firebase_init import db
from firebase_admin import firestore
import pandas as pd
import re
import unicodedata

def hotel_email(hotel_name, domain="example.com"):
    # Normalize unicode (é → e)
    name = unicodedata.normalize("NFKD", hotel_name)
    name = name.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    name = name.lower()

    # Replace any non-alphanumeric with a dot
    name = re.sub(r"[^a-z0-9]+", ".", name)

    # Trim leading/trailing dots
    name = name.strip(".")

    return f"{name}@{domain}"

def fill_dataset():
    df= pd.read_excel("All Sabre GDS Properties with Global Ids (Active)_Aug2025_2.xlsx", sheet_name="Page1_2")
    for index,row in df.iterrows():
        
        # Accesarea datelor:
        user_id = row['Sabre Property ID']
        hotel_name = row['Sabre Property name']
        adress = row['Address line 1']
        phone = row['Property Phone Number']
        nume_hotel_nou = f"Hotel_{user_id}"
        create_new_hotel_instance(
            hotel_name=nume_hotel_nou,
            details = {
                'user_id': user_id,
                'hotel_name': hotel_name,
                'adress': adress,
                'phone': phone,
                'email_address': hotel_email(hotel_name, domain="example.com"),
            }
        )

def create_new_hotel_instance(hotel_name, details=None):
    """
    Creează sau suprascrie un document în colecția 'hotels' folosind hotel_name ca ID.
    """
    if db is None:
        print("Nu se poate scrie, clientul Firestore nu este inițializat.")
        return False
        
    try:
        doc_ref = db.collection('hotels').document(hotel_name)
        
        data_to_set = details if details is not None else {
            'nume': hotel_name,
            'adresa': 'Adresă de test',
            'camere': 0,
            'data_creare': firestore.SERVER_TIMESTAMP 
        }
        
        # set() creează documentul cu ID-ul specificat
        doc_ref.set(data_to_set)
        
        print(f"\n✅ Instanță hotel creată/actualizată! ID: {hotel_name}")
        return True
        
    except Exception as e:
        print(f"❌ Eroare la crearea instanței hotelului: {e}")
        return False

def read_first_n_hotels(limit=5):
    """
    Fetches and prints the first N documents from the 'hotels' collection.
    """
    if db is None:
        print("❌ Cannot read: Firestore client is not initialized.")
        return

    print(f"\n--- Printing First {limit} Hotel Records from Firestore ---")
    try:
        # Use .limit() to only retrieve a few documents for inspection
        hotels_ref = db.collection('hotels').limit(limit)
        docs = hotels_ref.stream()

        for doc in docs:
            print(f"** Document ID: {doc.id} **")
            print(f"  Hotel Name: {doc.to_dict().get('hotel_name', 'N/A')}")
            print(f"  Address: {doc.to_dict().get('adress', 'N/A')}")
            print(f"  Phone: {doc.to_dict().get('phone', 'N/A')}")
            print(f"  Email: {doc.to_dict().get('email_address', 'N/A')}")
            print("-" * 30)

    except Exception as e:
        print(f"❌ Error reading from Firestore: {e}")

if __name__ == "__main__":
    read_first_n_hotels()