from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch

def loadDocument(filename="2405.09818v1.pdf", pageNumber=0):
    # load a document and return page contents
    reader = PdfReader(filename)
    page = reader.pages[pageNumber]
    return page.extract_text()

# pageContent = loadDocument()
# print(pageContent)

def extractMetaData(filename="2405.09818v1.pdf"):
    reader = PdfReader(filename)
    numberOfPages = len(reader.pages)
    print("Length of the document =", numberOfPages)
    metadata = reader.metadata
    return metadata, numberOfPages

metaData, numberOfPages = extractMetaData()
# print(metaData, type(metaData))

documentContent = []
for page in range(numberOfPages - 1):
    pageContent = loadDocument(pageNumber=page)
    pageContent = pageContent.replace("-", "")
    pageContent = pageContent.replace("\n", " ").split(". ")
    documentContent += pageContent

print(len(documentContent), documentContent[:5])


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
# sentences = ["Questo è un esempio di frase", "Questo è un ulteriore esempio"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('efederici/sentence-bert-base')
model = AutoModel.from_pretrained('efederici/sentence-bert-base')

# Tokenize sentences
encoded_input = tokenizer(documentContent[:5], padding=True, truncation=True,
                          return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
print(sentence_embeddings.shape)

query = "Chameleon"

# Tokenize sentences
encodedQuery = tokenizer(query, padding=True, truncation=True,
                          return_tensors='pt')
# Compute token embeddings
with torch.no_grad():
    modelOutput = model(**encodedQuery)

queryEmbedding = mean_pooling(modelOutput, encodedQuery['attention_mask'])

similarities = torch.cosine_similarity(queryEmbedding, sentence_embeddings, dim=1)
top2Results = torch.topk(similarities, k=2)
print(similarities)
print(top2Results)

print("Similar sentences:")
print(f"Query : {query}")
top2Indices = top2Results.indices.tolist()
for i in top2Indices:
    print(documentContent[i])