import sys
sys.path.append('..')

from PyPDF2 import PdfReader

def Parser( pdfFile = '../papers/2301.03669.pdf'):
    # First task is to convert the pdf file to a text file.
    reader = PdfReader(pdfFile)
    content = ""
    
    for page in reader.pages:
        content += page.extract_text()
    
    with open(pdfFile.replace('.pdf', '.txt'), 'w') as f:
        f.write(content)

    return content


if __name__ == "__main__":
    Parser()
