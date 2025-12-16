import time
from interactor import VoiceAgent
from firebase_init import initialize_firebase_app

def test_logic():
    # 1. Initialize DB Connection
    db = initialize_firebase_app()
    hotel_id = "Hotel_1"
    
    # 2. GET INITIAL STATE (The correct Firestore way)
    print("--- FETCHING INITIAL DB STATE ---")
    doc_ref = db.collection("users").document(hotel_id)
    initial_snapshot = doc_ref.get()
    
    if not initial_snapshot.exists:
        print(f"❌ Error: Document {hotel_id} not found in Firestore. Run force_seed.py first.")
        return

    initial_data = initial_snapshot.to_dict()
    print(f"[DB STATE BEFORE] {initial_data}")

    # 3. INITIALIZE AGENT
    print("\n--- 1. INITIALIZING AGENT ---")
    agent = VoiceAgent(hotel_id=hotel_id)
    
    # 4. SIMULATE CONVERSATION
    # We simulate a user correcting their email
    user_input = "Actually, my email is denis@test.com"
    print(f"\n[USER SAYS]: {user_input}")

    # Run the agent logic
    response = agent.process_llm(user_input)
    print(f"[AGENT SAYS]: {response}")

    # 5. VERIFY UPDATE (Fetch fresh data from Cloud)
    print("\n--- VERIFYING UPDATE ---")
    
    # We must fetch the document AGAIN to see changes
    final_snapshot = doc_ref.get()
    final_data = final_snapshot.to_dict()
    
    print(f"[DB STATE AFTER] {final_data}")
    
    # 6. RUN TEST ASSERTION
    # safe get + lowercase comparison
    stored_email = final_data.get('email', '').lower()
    
    if stored_email == "denis@test.com":
        print("\n✅ TEST PASSED: Database was successfully updated in the Cloud!")
    else:
        print(f"\n❌ TEST FAILED: Expected 'denis@test.com', found '{final_data.get('email')}'")

if __name__ == "__main__":
    test_logic()