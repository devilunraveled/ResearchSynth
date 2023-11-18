import time
import numpy as np
from MatchSummarizer import MatchSummarizer
import sys
sys.path.append('..')
from Datasets.DataLoader import DataLoader

# loading the datasets
datasetLoader = DataLoader(datasetName='arxiv')

arxiv_test = datasetLoader.getData('../Datasets/', split='test')
datasetLoader.datasetName = 'pubmed'
pubmed_test = datasetLoader.getData('../Datasets/', split='test')

# pick only the first 1000 rows from the dataframes
arxiv_test = arxiv_test[:100]
pubmed_test = pubmed_test[:100]

# creating 'Gold Summary' column
def mapping(row):
    row['Gold Summary'] = ''.join(row['abstract_text'])
    return row

arxiv_test = arxiv_test.apply(mapping, axis=1)
pubmed_test = pubmed_test.apply(mapping, axis=1)

# loading the model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model2 = SentenceTransformer('ArtifactAI/arxiv-distilbert-base-v3-GenQ')
n_gram_size = 2
summary_size = 10

# generating summaries
def generateSummary(row):
    article = ''.join(row['article_text'])
    summarizer = MatchSummarizer(article, n_gram_size, model)
    summary = summarizer.generateSummary(summary_size)
    row['Generated Summary'] = summary
    print(f"Generated summary for {row['article_id']}.")
    return row

start_time = time.process_time()
arxiv_test = arxiv_test.apply(generateSummary, axis=1)
print('Time taken for arxiv: ', time.process_time() - start_time)

start_time = time.process_time()
pubmed_test = pubmed_test.apply(generateSummary, axis=1)
print('Time taken for pubmed: ', time.process_time() - start_time)

# evaluating the summaries
from Evaluation.evaluation import rougeScores
arxiv_test, rougeScoresArxiv = rougeScores(arxiv_test)
pubmed_test, rougeScoresPubmed = rougeScores(pubmed_test)

# printing the results
print('arxiv')
print('rouge1: ', np.mean([ score.fmeasure for score in rougeScoresArxiv['rouge1'] ]))
print('rouge2: ', np.mean([ score.fmeasure for score in rougeScoresArxiv['rouge2'] ]))
print('rougeL: ', np.mean([ score.fmeasure for score in rougeScoresArxiv['rougeL'] ]))
print('pubmed')
print('rouge1: ', np.mean([ score.fmeasure for score in rougeScoresPubmed['rouge1'] ]))
print('rouge2: ', np.mean([ score.fmeasure for score in rougeScoresPubmed['rouge2'] ]))
print('rougeL: ', np.mean([ score.fmeasure for score in rougeScoresPubmed['rougeL'] ]))

# saving the results
arxiv_test.to_csv('arxiv_test_matchSum.csv')
pubmed_test.to_csv('pubmed_test_matchSum.csv')
