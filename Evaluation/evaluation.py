from Scorer import Score
from evaluate import load
bertScore = load('bertscore')

def rougeScores(df):
    '''
    Input:
        Dataframe with the following mandatory columns: 'Gold Summary', 'Generated Summary'.
    Returns:
        Dataframe with the following additional columns: 'rouge1'.
        rougeScores: dictionary with the following keys: 'rouge1', 'rouge2', 'rougeL'. 
                     Each key has a list of corresponding rouge scores for each generated summary.
    '''
    rougeScores = {'rouge1': [],
                   'rouge2': [],
                   'rougeL': []}

    def mapping(row):
        trueSummary = row['Gold Summary']
        predSummary = row['Generated Summary']

        score = Score(trueSummary, predSummary)
        rougeScores['rouge1'].append(score.rougeScore()['rouge1'])
        rougeScores['rouge2'].append(score.rougeScore()['rouge2'])
        rougeScores['rougeL'].append(score.rougeScore()['rougeL'])

        row['rouge1'] = score.rougeScore()['rouge1'].fmeasure
        row['rouge2'] = score.rougeScore()['rouge2'].fmeasure
        row['rougeL'] = score.rougeScore()['rougeL'].fmeasure

        results = bertScore.compute(predictions=[predSummary],
                                    references=[trueSummary],
                                    lang='en')
        row['bertScore'] = results['f1'][0]
        return row
    
    df = df.apply(mapping, axis=1)

    return df, rougeScores
