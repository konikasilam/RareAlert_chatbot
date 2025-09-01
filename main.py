import random
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator


df = pd.read_csv(r'D:\healthcareChatbot\rarealert\rare_diseases_50.csv') 


model = SentenceTransformer('all-MiniLM-L6-v2')


fallback_responses = {
    "rash": "Rashes may be symptoms of rare skin conditions. Please consult a dermatologist.",
    "weakness": "This might relate to metabolic disorders. Kindly consider seeing a neurologist.",
    "vision": "Vision issues can relate to genetic disorders. Visit an ophthalmologist immediately.",
    "pain": "Persistent unexplained pain can be a symptom of a rare nerve condition.",
}



rare_health_tips = {
    "genetic": [
        "Genetic counseling can help you understand inherited conditions.",
        "Regular check-ups are vital if your family has a history of rare conditions.",
    ],
    "autoimmune": [
        "Autoimmune diseases may flare unpredictably. Maintain a healthy diet.",
        "Avoid stress and get adequate sleep to reduce autoimmune triggers.",
    ],
    "neurological": [
        "If symptoms like muscle weakness or numbness persist, consult a neurologist.",
        "Physical therapy can help manage some rare neurological disorders.",
    ],
    "general": [
        "Track symptoms and maintain a health diary to share with specialists.",
        "Join rare disease support groups for information and emotional support.",
    ]
}



def get_rare_health_tip(user_input):
    text = user_input.lower()
    if "genetic" in text or "inherited" in text:
        return random.choice(rare_health_tips["genetic"])
    elif "immune" in text or "autoimmune" in text:
        return random.choice(rare_health_tips["autoimmune"])
    elif "numb" in text or "muscle" in text or "neurological" in text:
        return random.choice(rare_health_tips["neurological"])
    else:
        return random.choice(rare_health_tips["general"])




def predict_rare_disease(user_input):
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    disease_embeddings = model.encode(df['Symptoms'].tolist(), convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(user_input_embedding, disease_embeddings)[0]
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()

    SIMILARITY_THRESHOLD = 0.5  # You can tweak this

    if best_match_score < SIMILARITY_THRESHOLD:
        for keyword, fallback in fallback_responses.items():
            if keyword in user_input.lower():
                return fallback
        return "Sorry, I couldn't identify the disease. Please consult a specialist."

    match = df.iloc[best_match_idx]
    return f"Possible Rare Disease: {match['Disease']}\nRecommended Remedy: {match['Remedy']}\nSpecialist Doctor: {match['Specialist']}"




# Translate text
def translate_text(text, dest_language='en'):
    return translator.translate(text, dest=dest_language).text


# Initialize translator
translator = Translator()

# Streamlit UI
st.set_page_config(page_title="Rare Alert - Rare Disease Assistant")
st.title("🩺 Rare Alert - Rare Disease Chatbot")

user_input = st.text_input("Describe your symptoms:")

language_choice = st.selectbox("Select Language", [
    "English", "Hindi", "Gujarati", "Korean", "Turkish",
    "German", "French", "Arabic", "Urdu", "Tamil", "Telugu", "Chinese", "Japanese"
])

language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Korean": "ko",
    "Turkish": "tr",
    "German": "de",
    "French": "fr",
    "Arabic": "ar",
    "Urdu": "ur",
    "Tamil": "ta",
    "Telugu": "te",
    "Chinese": "zh-CN",
    "Japanese": "ja",
}

# Diagnosis button
if st.button("Diagnose"):
    if user_input:
        diagnosis = predict_rare_disease(user_input)
        translated = translate_text(diagnosis, dest_language=language_codes[language_choice])
        st.success(f"🩻 Diagnosis: {translated}")
        st.info("*Note: For accurate diagnosis, please consult a medical professional.*")

# Health tip button
if st.button("Get Health Tip"):
    if user_input:
        tip = get_rare_health_tip(user_input)
        translated_tip = translate_text(tip, dest_language=language_codes[language_choice])
        st.write(f"💡 **Health Tip:** {translated_tip}")
        st.caption("*Note: Tips are AI-generated and for general awareness.*")