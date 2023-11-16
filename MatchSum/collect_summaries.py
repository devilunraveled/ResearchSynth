from MatchSummarizer import MatchSummarizer
from Datasets.DataLoader import DataLoader

# loading the datasets
datasetLoader = DataLoader(datasetName='arxiv')

arxiv_test = datasetLoader.getData('../Datasets/', split='test')
datasetLoader.datasetName = 'pubmed'
pubmed_test = datasetLoader.getData('../Datasets/', split='test')

# creating 'Gold Summary' column
def mapping(row):
    row['Gold Summary'] = ''.join(row['abstract_text'])
    return row

arxiv_test = arxiv_test.apply(mapping, axis=1)
pubmed_test = pubmed_test.apply(mapping, axis=1)

# loading the model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
n_gram_size = 7
summary_size = 8

# generating summaries
def generateSummary(row):
    article = ''.join(row['article_text'])
    summarizer = MatchSummarizer(article, n_gram_size, model=model)
    summary = summarizer.generateSummary(summary_size)
    row['Generated Summary'] = summary
    return row

arxiv_test = arxiv_test.apply(generateSummary, axis=1)
pubmed_test = pubmed_test.apply(generateSummary, axis=1)

# evaluating the summaries
from Evaluation.evaluation import rougeScores
arxiv_test, rougeScoresArxiv = rougeScores(arxiv_test)
pubmed_test, rougeScoresPubmed = rougeScores(pubmed_test)

# printing the results
print('arxiv')
print('rouge1: ', sum(rougeScoresArxiv['rouge1'])/len(rougeScoresArxiv['rouge1']))
print('rouge2: ', sum(rougeScoresArxiv['rouge2'])/len(rougeScoresArxiv['rouge2']))
print('rougeL: ', sum(rougeScoresArxiv['rougeL'])/len(rougeScoresArxiv['rougeL']))
print('pubmed')
print('rouge1: ', sum(rougeScoresPubmed['rouge1'])/len(rougeScoresPubmed['rouge1']))
print('rouge2: ', sum(rougeScoresPubmed['rouge2'])/len(rougeScoresPubmed['rouge2']))
print('rougeL: ', sum(rougeScoresPubmed['rougeL'])/len(rougeScoresPubmed['rougeL']))

# saving the results
arxiv_test.to_csv('arxiv_test_matchSum.csv')
pubmed_test.to_csv('pubmed_test_matchSum.csv')
