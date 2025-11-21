import pandas as pd
import io
from docx import Document
import PyPDF2
import streamlit as st

def parse_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == 'text/plain':
            text = uploaded_file.read().decode('utf-8')
        elif uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            text = ' '.join(df.astype(str).values.flatten())
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            text = ' '.join([para.text for para in doc.paragraphs])
        elif uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
    return text
