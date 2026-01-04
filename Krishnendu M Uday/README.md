# NarrativeNexus

Small Streamlit app to upload and list text sample files.

## Setup (Windows, cmd.exe)

Open a terminal in the `NarrativeNexus` folder and run:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Setup (Linux / macOS)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

## Tests & Linting

Run tests with pytest (from the project folder):

```bat
venv\Scripts\activate
pytest -q
```

Run flake8 to lint the code:

```bat
venv\Scripts\activate
flake8
```
