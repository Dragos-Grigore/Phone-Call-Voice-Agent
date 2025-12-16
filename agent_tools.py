from firebase_init import initialize_firebase_app
from google.cloud import firestore
from fill_dataset import read_first_n_hotels

db = initialize_firebase_app()


def get_hotel_info(user_id: str) -> dict:
    """Get the current hotel's information from Firestore"""
    
    # Linter is now happy because of the assert above
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()

    if not doc.exists:
        return {}

    data = doc.to_dict()
    if not data:
        return {}

    # Return using safe .get() calls
    return {
        "hotel_name": data.get("hotel", data.get("hotel_name")),
        "phone": data.get("phone_number", data.get("phone")),
        "email": data.get("email"),
        "address": data.get("address")
    }
    
def verify_with_hotel(current_db_data: dict, extracted_hotel_data: dict) -> dict:
    """
    Compare the data provided by the hotel (extracted from voice) 
    with the current information from the database.
    """
    updates = {}

    # Loop through the keys we currently have in the DB
    for key, db_val in current_db_data.items():
        if key in extracted_hotel_data:
            new_val = extracted_hotel_data[key]

            # Compare safely (handle None and mismatched types)
            str_db = str(db_val).strip().lower() if db_val else ""
            str_new = str(new_val).strip().lower() if new_val else ""

            if str_db != str_new:
                updates[key] = new_val

    return {
        "needs_update": bool(updates),
        "updates": updates
    }


def update_data(hotel_id: str, updated_info: dict):
    """
    Update the database if needed (Firestore version).
    """
    try:
        # Use Firestore update syntax
        doc_ref = db.collection("users").document(hotel_id)
        doc_ref.update(updated_info)
        return True
    except Exception as e:
        print(f"❌ Firestore Update Error: {e}")
        return False

read_first_n_hotels()