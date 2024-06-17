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

if __name__ == '__main__':
    metaData, numberOfPages = extractMetaData()
    # print(metaData, type(metaData))

    documentContent = []
    pageNumbers = []
    for page in range(numberOfPages - 1):
        pageContent = loadDocument(pageNumber=page)
        pageContent = pageContent.replace("- ", "")
        pageContent = pageContent.replace("\n", " ").split(". ")
        documentContent += pageContent
        pageNumbers += [page + 1] * len(pageContent)

    print(len(documentContent), pageNumbers[-5:],
          documentContent[-5:])


    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('efederici/sentence-bert-base')
    model = AutoModel.from_pretrained('efederici/sentence-bert-base')

    # Tokenize sentences
    encodedInput = tokenizer(documentContent, padding=True, truncation=True,
                             return_tensors='pt')
    device = "mps"
    if torch.cuda.is_available():
        device = "cuda"

    model.to(device)
    # Compute token embeddings
    with torch.no_grad():
        encodedInput = encodedInput.to(device)
        modelOutput = model(**encodedInput)

    # Perform pooling. In this case, mean pooling.
    sentenceEmbeddings = mean_pooling(modelOutput, encodedInput['attention_mask'])

    print("Sentence embeddings:")
    # print(sentenceEmbeddings)
    print(sentenceEmbeddings.shape)

    query = "Chameleon"

    # Tokenize sentences
    encodedQuery = tokenizer(query, padding=True, truncation=True,
                              return_tensors='pt')
    encodedQuery = encodedQuery.to(device)
    # Compute token embeddings
    with torch.no_grad():
        modelOutput = model(**encodedQuery)

    queryEmbedding = mean_pooling(modelOutput, encodedQuery['attention_mask'])

    similarities = torch.cosine_similarity(queryEmbedding, sentenceEmbeddings, dim=1)
    topKResults = torch.topk(similarities, k=5)
    print(topKResults)

    print("Similar sentences:")
    print(f"Query : {query}")
    topKIndices = topKResults.indices.tolist()
    for i in topKIndices:
        print(f"i = {i}")
        print(documentContent[i])
        print()