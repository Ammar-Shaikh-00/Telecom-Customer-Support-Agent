import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# Load .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
print(f"🔍 Looking for .env at: {env_path}")

load_dotenv(dotenv_path=env_path)
print("OPENAI_API_KEY from env file:", os.getenv("OPENAI_API_KEY"))

# Debug: print all env keys (not values, for safety)
print("✅ Loaded ENV keys:", list(os.environ.keys()))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY missing in .env (check spelling/encoding)")
else:
    print("✅ OPENAI_API_KEY loaded correctly")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# ✅ Local embeddings model (free to use)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Expanded Telecom FAQs
telecom_faqs = [
    {"question": "How can I recharge my mobile balance?", 
     "answer": "You can recharge via scratch card, mobile app, or USSD code *123#."},

    {"question": "How to check remaining data balance?", 
     "answer": "Dial *456# or use the telecom app to check your remaining MBs."},

    {"question": "What should I do if my SIM is lost?", 
     "answer": "Call our helpline immediately or visit the nearest franchise to block your SIM."},

    {"question": "How to activate international roaming?", 
     "answer": "International roaming can be activated from the app or by calling the helpline."},

    {"question": "How to contact customer support?", 
     "answer": "You can dial 111 from your SIM or use live chat in our app."},

    {"question": "How to check my call and SMS packages?", 
     "answer": "Dial *707# or open the mobile app to check your active call and SMS packages."},

    {"question": "How to subscribe to an internet package?", 
     "answer": "Visit the telecom app or dial *345# to subscribe to available internet bundles."},

    {"question": "How to pay my postpaid bill?", 
     "answer": "You can pay your postpaid bill online via debit/credit card, banking apps, or at retail outlets."},

    {"question": "What should I do if my number is blocked?", 
     "answer": "Please clear outstanding dues or visit our franchise to reactivate your number."},

    {"question": "How do I upgrade my SIM to 4G or 5G?", 
     "answer": "Visit the nearest customer service center with your CNIC to upgrade your SIM."},

    {"question": "How can I share balance with another number?", 
     "answer": "Use the balance share service by dialing *100*recipient-number*amount#."},

    {"question": "How to register a complaint?", 
     "answer": "Dial 111 or use the complaint section in our mobile app to register service complaints."},

    {"question": "How to check my call history?", 
     "answer": "Call records can be accessed through the telecom app or by visiting our customer care office."},

    {"question": "What should I do if my internet is too slow?", 
     "answer": "Try restarting your phone, check your data limit, or contact customer support for assistance."},

    {"question": "How can I activate caller tune service?", 
     "answer": "You can activate caller tunes via the telecom app or by sending an SMS with your chosen tune code."},

    {"question": "How can I deactivate promotional messages?", 
     "answer": "Send 'STOP' to 9999 to opt out of promotional SMS services."},

    {"question": "How to transfer ownership of a SIM?", 
     "answer": "Both current and new owners must visit a franchise with their CNICs for SIM transfer."},

    {"question": "How do I activate Do Not Disturb (DND) service?", 
     "answer": "Dial *362# or use the app to enable the DND service and block unwanted calls/messages."},

    {"question": "How to check my current tariff plan?", 
     "answer": "Dial *101# or log in to the mobile app to view your active tariff plan."}
]

# Precompute embeddings for all FAQ questions
faq_questions = [faq["question"] for faq in telecom_faqs]
faq_embeddings = embedder.encode(faq_questions, convert_to_tensor=True)


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    # Embed user question
    user_embedding = embedder.encode(query.question, convert_to_tensor=True)

    # Find closest FAQ
    similarities = util.pytorch_cos_sim(user_embedding, faq_embeddings)[0]
    best_match_idx = int(similarities.argmax())
    best_faq = telecom_faqs[best_match_idx]

    # If similarity is high, return FAQ directly
    if similarities[best_match_idx] > 0.65:
        return {"answer": best_faq["answer"]}

    # Else ask OpenAI (fallback)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a telecom customer support assistant."},
            {"role": "user", "content": query.question}
        ]
    )

    return {"answer": completion.choices[0].message.content.strip()}
