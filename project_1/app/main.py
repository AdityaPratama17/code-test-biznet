import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# load model
if "clf" not in st.session_state:
    st.session_state['clf'] = joblib.load('resources/model/trained_model.pkl')
    st.session_state['bert_model'] = SentenceTransformer('all-MiniLM-L6-v2')


def predict(input_text:str):
    # Lakukan embedding text
    embedding = st.session_state.bert_model.encode([input_text])

    # Prediksi label
    label_int = st.session_state.clf.predict(embedding)[0]
    label_prob = st.session_state.clf.predict_proba(embedding)[0]

    # Mapping angka ke label string
    label_str = {
        0: 'information',
        1: 'request',
        2: 'problem'
    }.get(label_int, 'unknown')

    return label_str, label_prob

# title
st.set_page_config(page_title="Customer Message Classifier", layout="centered")
st.title("📩 Customer Message Classifier")
st.markdown("Masukkan pesan pelanggan di bawah untuk diklasifikasikan ke dalam kategori: **information**, **request**, atau **problem**.")

# Input user
text_input = st.text_area("Tulis pesan pelanggan di sini:", height=100)

# Tombol prediksi
if st.button("🔍 Prediksi"):
    if text_input.strip() == "":
        st.warning("Mohon masukkan pesan terlebih dahulu.")
    
    else:
        # predict text
        label, label_prob = predict(text_input)

        st.markdown(f"### Hasil Prediksi : {label}")
        container = st.container(border=True)
        with container:
            st.markdown(f"""
                ##### Prob score:
                - **information:** {round(label_prob[0]*100, 2)}%
                - **request:** {round(label_prob[1]*100, 2)}%
                - **problem:** {round(label_prob[2]*100, 2)}%
            """)
