from firebase_init import db
from firebase_admin import firestore
import time 
import pandas as pd

def fill_dataset():
    df= pd.read_excel("All Sabre GDS Properties with Global Ids (Active)_Aug2025_2.xlsx", sheet_name="Page1_1")
    for index,row in df.iterrows():
        
        # Accesarea datelor:
        hotel_id = row['Sabre Property ID']
        hotel_name = row['Sabre Property name']
        adress1 = row['Address line 1']
        adress2 = row['Address line 2']
        city = row['City']
        state = row['State']
        zip = row['Zip']
        country = row['Country Code']
        phone = row['Property Phone Number']
        fax = row['Property Fax Number']
        airport = row['Primary Airport Code']
        rating = row['Sabre Property Rating']
        latitude = row['Property Latitude']
        longitude = row['Property Longitude']
        chain = row['Chain code']
        source = row['Source Code']
        global_id = row['Global Property ID']
        nume_hotel_nou = f"Hotel_Creat_{hotel_id}"
        create_new_hotel_instance(
            hotel_name=nume_hotel_nou,
            details={
                'hotel_id':hotel_id,
                'hotel_name': hotel_name,
                'adress1': adress1,
                'adress2': adress2,
                'city': city,
                'state': state,
                'zip': zip,
                'country': country,
                'phone': phone,
                'fax':fax,
                'airport':airport,
                'rating':rating,
                'latitude':latitude,
                'longitude':longitude,
                'chain':chain,
                'source':source,
                'global_id':global_id 
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

if __name__ == "__main__":
    fill_dataset()