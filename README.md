# HomeMatch: use of llm and vector database to generate and search real estate ads

## Description
Through the `langchain` library, this project exploits an LLM to generate real estate ads including house descriptions, and to display results optimized according to the user's request.

## Installation
After retrieving the project, it is necessary to create a virtual environment and install the required dependencies.
```bash
pip install -r requirements.txt
```

A `.env` file must then be created at the root of the project, containing the following environment variables:
```bash
OPENAI_API_KEY=...
OPENAI_API_BASE=https://...
```

## Usage
Firstly, via the Jupyter python notebook `HomeMatch.ipynb`, to generate real estate ads and store them in a vector database, then test the functionality and ad augmentation.

The ads generated are also stored in a txt file `listings.txt` in JSON format (and also in `listings.jsonl`), for later use (including with `HomeMatch.py`).

Via gradio-based application to provide a user interface for searching the vector database and displaying results:
```bash
python HomeMatch.py
```
Apart from the user interface, this script takes over all the code relating to the search and ad increase functionalities present in the Jupyter `HomeMatch.ipynb` notebook.