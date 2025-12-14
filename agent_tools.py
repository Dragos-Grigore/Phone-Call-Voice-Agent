from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
#from firebase_init import db

db = {
    "user_1": {
        "hotel_name": "Hotel Aurora",
        "phone": "+39 123 456 789",
        "email": "info@aurora.it",
        "address": "Via Roma 10"
    },
    "user_2": {
        "hotel_name": "Hotel Bella",
        "phone": "+39 987 654 321",
        "email": "contact@bella.it",
        "address": "Via Milano 20"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_hotel_info(runtime: ToolRuntime[UserContext]) -> dict:
    """"Get the current hotel's information from database"""
    user_id = runtime.context.user_id
    user_ref = db.reference(f"users/{user_id}").get()

    return {
        "hotel_name": user_ref.get("hotel"),
        "phone": user_ref.get("phone_number"),
        "email": user_ref.get("email"),
        "address": user_ref.get("address")
    }
    
@tool
def verify_with_hotel(db_data: dict, hotel_data : dict) -> dict:
    """
    Compare the data provided by the hotel with the information from the database.
    Return only the fields that need to be updated.
    """
    updates = {}

    for key, db_val in db.data.items():
        hotel_val = hotel_data.get(key)

        if  str(db_val).strip().lower() != str(hotel_val).strip().lower():
            updates[key] = hotel_val

    return {
        "needs_update" : bool(updates),
        "updates" : updates
            }


@tool
def update_data(runtime: "ToolRuntime[UserContext]", updated_info: dict):
    """
    Update the database if needed.
    Return a succes/error mesage.
    """
    user_id = runtime.context.user_id
    user_ref = db.reference(f"users/{user_id}")
    user_ref.update(updated_info)
