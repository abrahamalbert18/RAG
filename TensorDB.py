import torch
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import ProcessDocument

class TensorDB():
    def __init__(self, documentName):
        self.documentName = documentName
        self.metadata, self.numberOfPages = self.extractMetaData()
        self.document = None
        self.createDocumentDataFrame()
        pass

    def extractMetaData(self):
        metaData, numberOfPages = ProcessDocument.extractMetaData(
                                                self.documentName)
        return metaData, numberOfPages

    def loadPageContent(self, page):
        pageContent = ProcessDocument.loadDocument(self.documentName,
                                                   pageNumber=page)
        return pageContent

    def loadDocumentContent(self):
        documentContent = []
        pageNumbers = []
        for page in range(self.numberOfPages - 1):
            pageContent = self.loadPageContent(page=page)
            pageContent = pageContent.replace("- ", "")
            pageContent = pageContent.replace("\n", " ").split(". ")
            documentContent += pageContent
            pageNumbers += [page + 1] * len(pageContent)
        return documentContent, pageNumbers

    def createDocumentDataFrame(self):
        documentContent, pageNumbers = self.loadDocumentContent()
        self.document = pd.DataFrame({"pageNumber" : pageNumbers,
                        "line" : documentContent})
        pass

    def saveDocument(self, path="./DocumentDatabases",
                     filename="Chameleon"):
        assert "embedding" in self.document.columns
        self.document.to_parquet(f"{path}/{filename}")
        pass

    def update(self):
        pass

    def add(self, documentTensor):
        pass

    def reset(self):
        self.document = None
        pass

if __name__ == '__main__':
    data = TensorDB(documentName="2405.09818v1.pdf")
    print(data.document)