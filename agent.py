from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# LLM (local or HF-hosted)
# -----------------------------
model_id = "google/flan-t5-base"  # instruction-following, deterministic

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -----------------------------
# Prompt
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer ONLY using the context. "
        "If the answer is not in the context, say 'Not found in context.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

qa_runnable = prompt | llm

context_text = """
What hotel name?
The hotel is officially called The Grand Hilton, though most people just say Hilton.

What address?
It's located at 42 X Street, right across from the central park.

What email address?
Their booking email is reservations@hilton-grand.com.
"""


result_name = qa_runnable.invoke({
    "context": context_text,
    "question": "What is the name of the hotel?"
})

result_address = qa_runnable.invoke({
    "context": context_text,
    "question": "What is the address of the hotel?"
})

result_email = qa_runnable.invoke({
    "context": context_text,
    "question": "What is the email address of the hotel?"
})

print(result_name)
print(result_address)
print(result_email)