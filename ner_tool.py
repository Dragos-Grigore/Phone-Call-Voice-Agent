# ner_tool.py
import re
from langchain.agents import Tool

def extract_entities_from_dialog(dialog: str):
    lines = [l.strip() for l in dialog.split("\n") if l.strip()]
    entities = {
        "hotel_name": None,
        "address": None,
        "email": None,
    }

    last_question = None

    for line in lines:
        lower = line.lower()

        # Detectare întrebări
        if "cum se numește hotel" in lower or "cum se numeste hotel" in lower:
            last_question = "hotel_name"
            continue

        if "ce adresa are" in lower or "care este adresa" in lower:
            last_question = "address"
            continue

        if "ce email" in lower or "adresa de email" in lower:
            last_question = "email"
            continue

        # Detectare răspuns
        if last_question == "hotel_name":
            entities["hotel_name"] = line
        elif last_question == "address":
            entities["address"] = line
        elif last_question == "email":
            entities["email"] = line

        last_question = None

    # Regex email fallback
    if entities["email"] is None:
        match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", dialog)
        if match:
            entities["email"] = match.group(0)

    return str(entities)


ner_tool = Tool(
    name="dialog_ner",
    description="Extrage numele hotelului, adresa și email dintr-un dialog liber.",
    func=extract_entities_from_dialog
)
