import torch
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import ProcessDocument
import os

class TransformerModel:
    def __init__(self, modelName='efederici/sentence-bert-base',
                 device="cpu"):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.model = AutoModel.from_pretrained(modelName)
        self.device = device
        self.query = "Chameleon"
        self.topKResults = None # will be replaced by K-shaped tensor
        if device != "cpu":
            self.updateDevice()
        pass

    def updateDevice(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "mps" # Apple Silicon
        self.tokenizer.to(self.device)
        self.model.to(self.device)
        pass

        # Mean Pooling - Take attention mask into account for correct averaging

    def meanPooling(self, modelOutput, attentionMask):
        # First element of model_output contains all token embeddings
        tokenEmbeddings = modelOutput[0]
        inputMaskExpanded = attentionMask.unsqueeze(-1).expand(
                                tokenEmbeddings.size()).float()
        return torch.sum(tokenEmbeddings * inputMaskExpanded,
                         1) / torch.clamp(inputMaskExpanded.sum(1), min=1e-9)

    def encodeDocument(self, documentContent=""):
        # Tokenize sentences
        encodedDocument = self.tokenizer(documentContent, padding=True,
                                  truncation=True, return_tensors='pt')
        return encodedDocument

    def encodeAndEmbedQuery(self, query=""):
        self.query = query
        # Tokenize sentences
        encodedQuery = self.tokenizer(query, padding=True, truncation=True,
                                        return_tensors='pt')
        encodedQuery = encodedQuery.to(self.device)
        embeddedQuery = self.computeSentenceEmbeddings(encodedQuery)
        return embeddedQuery

    def computeSentenceEmbeddings(self, encodedInput):
        """
        Can be used for computing sentence embedding for both
        document and query
        :param encodedInput: String or List of Strings
        :return: sentence embeddings tensor of shape (len(encodedInput), 768)
        """
        # Compute token embeddings
        with torch.no_grad():
            encodedInput = encodedInput.to(self.device)
            modelOutput = self.model(**encodedInput)

        # Perform pooling. In this case, mean pooling.
        sentenceEmbeddings = self.meanPooling(modelOutput,
                                              encodedInput["attention_mask"])
        return sentenceEmbeddings

    def computeSimilarities(self, queryEmbedding,
                            documentEmbeddings, k=5):
        similarities = torch.cosine_similarity(queryEmbedding,
                                               documentEmbeddings, dim=1)
        topKResults = torch.topk(similarities, k=k)
        # print(topKResults)
        return topKResults

    def printResults(self, documentContent=None):
        print("Similar sentences:")
        print(f"Query : {self.query}")
        topKIndices = self.topKResults.indices.tolist()
        for i in topKIndices:
            print(f"i = {i}")
            print(documentContent[i])
            print()

class TensorDB(TransformerModel):
    def __init__(self, documentName):
        super().__init__(modelName='efederici/sentence-bert-base',
                        device="cpu")
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
        self.update()
        pass

    def saveDocument(self, path="./DocumentDatabases",
                     filename="Chameleon"):
        assert "embeddings" in self.document.columns
        self.document.to_parquet(f"{filename}")
        pass

    def update(self, dirName="Chameleon"):
        lines = self.document["line"].tolist()
        encodedLines = self.encodeDocument(lines)
        embeddedLines = self.computeSentenceEmbeddings(encodedLines)

        if not os.path.exists(dirName):
           os.mkdir(dirName)
        filePath = f"{dirName}/embeddings.pth.tar"
        torch.save(embeddedLines, filePath)
        self.document["embeddings"] = filePath
        pass

    def add(self, documentTensor):
        pass

    def reset(self):
        self.document = None
        pass


if __name__ == '__main__':
    data = TensorDB(documentName="2405.09818v1.pdf")
    print(data.document)