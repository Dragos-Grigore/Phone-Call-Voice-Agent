import re
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Load Model Once ---
print("Loading NER Model (Flan-T5)...")
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer ONLY using the context. "
        "If the answer is not in the context, say 'Not found'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

qa_runnable = prompt | llm

def extract_entities_from_dialog(text: str) -> dict:
    """
    Extracts hotel_name, address, email, phone from text.
    """
    entities = {}
    queries = {
        "hotel_name": "What is the name of the hotel?",
        "address": "What is the address of the hotel?",
        "email": "What is the email address?",
        "phone": "What is the phone number?"
    }

    for field, question in queries.items():
        try:
            res = qa_runnable.invoke({"context": text, "question": question})
            clean = res.strip()
            if "not found" not in clean.lower() and len(clean) > 2:
                entities[field] = clean
        except Exception as e:
            print(f"NER Error ({field}): {e}")

    return entities