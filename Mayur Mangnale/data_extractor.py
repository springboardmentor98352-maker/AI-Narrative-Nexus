import pandas as pd # type: ignore
from PyPDF2 import PdfReader # type: ignore
from io import StringIO


def extract_text_from_file(uploaded_file=None, pasted_text=None):

    if pasted_text and pasted_text.strip():
        return pasted_text, "txt", None, None

    if uploaded_file is None:
        return None, None, None, "No file uploaded or text pasted."

    filename = uploaded_file.name
    file_type = filename.split(".")[-1].lower()

    try:
        if file_type == "txt":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
            return text, "txt", None, None

        elif file_type == "pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text, "pdf", None, None

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            return None, "csv", df, None

        return None, None, None, "Unsupported file format."

    except Exception as e:
        return None, None, None, f"Error reading file: {str(e)}"
