from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. HuggingFacePipeline ---
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=gen_pipeline)

# --- 2. Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract hotel_name, address, and email from the dialog. Output JSON only."),
    ("user", "{input}")
])

# --- 3. LLMChain ---
chain = LLMChain(prompt=prompt, llm=llm)

# --- 4. Input ---
dialog = "What hotel name? The Grand Hilton\nWhat address? 42 X Street\nWhat email address? reservations@hilton-grand.com"

# --- 5. Run chain ---
output = chain.run(dialog)
print(output)
