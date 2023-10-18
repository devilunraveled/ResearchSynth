from rouge_score import rouge_scorer

class Score:
    def __init__(self, trueSummary = None, predSummary = None):
        self.trueSummary = trueSummary if trueSummary is not None else ""
        self.predSummary = predSummary if predSummary is not None else ""

    def rougeScore(self):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(self.trueSummary, self.predSummary)

        return scores
