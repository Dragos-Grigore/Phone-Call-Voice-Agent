db = {"Hotel_1": {
        "hotel_name": "The Hotel Chelsea",
        "adress": "222 West Twenty-Third Street, New York",
        "phone": "+40764067966",
        "email": "reservations@hetelcelsea.com"
    }
}

def get_hotel_info(user_id: str) -> dict:
    """"Get the current hotel's information from database"""
    user_ref = db.get(user_id, {}).copy()

    return {
        "hotel_name": user_ref.get("hotel"),
        "phone": user_ref.get("phone_number"),
        "email": user_ref.get("email"),
        "address": user_ref.get("address")
    }
    

def verify_with_hotel(db_data: dict, hotel_data: dict) -> dict:
    updates = {}
    
    # Simple direct matching now that keys are standardized
    for key, val in hotel_data.items():
        if key in db_data:
            current_val = db_data[key]
            # Compare ignoring case/whitespace
            if str(current_val).strip().lower() != str(val).strip().lower():
                updates[key] = val
    
    return {
        "needs_update": bool(updates),
        "updates": updates
    }



def update_data(user_id: str, updated_info: dict):
    """
    Update the database if needed.
    Return a succes/error mesage.
    """
    # user_ref = db.reference(f"users/{user_id}")
    # user_ref.update(updated_info)

    if user_id in db:
        db[user_id].update(updated_info)
