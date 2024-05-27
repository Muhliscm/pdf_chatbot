from pathlib import Path
from PyPDF2 import PdfReader


def pdf_reader(path: Path) -> str:
    """ 
    Args:
        path :pdf file path
    Returns
        str: string contains all pdf text
    """
    text = ""
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
