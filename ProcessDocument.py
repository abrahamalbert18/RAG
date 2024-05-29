from PyPDF2 import PdfReader

def loadDocument(filename="2405.09818v1.pdf", pageNumber=0):
    # load a document and return page contents
    reader = PdfReader(filename)
    page = reader.pages[pageNumber]
    return page.extract_text()

pageContent = loadDocument()
print(pageContent)