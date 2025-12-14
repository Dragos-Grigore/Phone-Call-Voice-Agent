from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

def remove_stopwords(text: str) -> str:
    tokens = re.findall(r"\b\w+\b", text)
    filtered = [t for t in tokens if t.lower() not in STOPWORDS]
    return " ".join(filtered)

model_id = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=16,
    do_sample=False,

)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

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
A: People keep mixing up the name.
B: Officially it carries the full “Grand” title under the Hilton brand.
A: Yet in practice?
B: Most shorten it to just Hilton when speaking casually.

A: And the location details?
B: Number 42 on X Street.
A: Any landmarks?
B: Directly facing the city’s main park—hard to miss if you know the area.

A: Booking contact still the same?
B: Yes, reservations go through their dedicated inbox.
A: Which one?
B: The reservations address tied to the hilton-grand domain.
"""


result_name = qa_runnable.invoke({
    "context": context_text,
    "question": "What is the name of the hotel?"
})

result_address = qa_runnable.invoke({
    "context": context_text,
    "question": "What is the address of the hotel>"
})

result_email = qa_runnable.invoke({
    "context": context_text,
    "question": "What is the email address of the hotel?"
})

print(result_name)
print(result_address)
print(result_email)
raw_address = result_address.strip()
clean_address = remove_stopwords(raw_address)
print(clean_address)