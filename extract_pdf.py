from PyPDF2 import PdfReader
import sys

pdf_path = sys.argv[1]
reader = PdfReader(pdf_path)
text = ''
for page in reader.pages:
    text += page.extract_text()
print(text)