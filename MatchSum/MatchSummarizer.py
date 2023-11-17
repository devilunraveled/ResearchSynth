import numpy as np

class MatchSummarizer:
    def __init__(self, paper, n_gram_size, model=None):
        '''
        paper: the research paper to summarize as a string.
        n_gram_size: the size of the n-grams to use for summarization.
        model: the model that will return the embedding for a given text.
               should have a method `encode` that takes a list of strings 
               and returns a list of embeddings.
        '''
        if model is None:
            raise ValueError("model cannot be None.")

        # get the sentences
        self.paper = paper
        sentences = paper.split(".")
        self.sentences = sentences
        self.model = model

        # generate n-grams
        n = n_gram_size
        self.n_gram_size = n_gram_size
        n_grams = self.generate_n_grams(n, sentences)
        self.n_grams = n_grams

        # calculate similarity of each n-gram with the paper
        sims = []
        for gram in n_grams:
            embeddings1 = self.model.encode([gram[0], paper], normalize_embeddings=True)
            sims.append((np.dot(embeddings1[0], embeddings1[1]), gram[1]))

        self.sims = sims

    def generate_n_grams(self, n, sentences): 
        '''
        returns an array n-grams where n-grams[i] = (ith n-gram, i)
        '''
        return [('.'.join(sentences[i:i+n]), i) for i in range(len(sentences)-n+1)]
    
    def generateSummary(self, summary_size):
        '''
        Generates a summary of approximatly `summary_size` sentences
        '''
        # select top k sentences
        k = summary_size
        sorted_n_grams = sorted(self.sims, key=lambda x: x[0], reverse=True)
        
        n_gram_ids = []
        done_set = set()

        # select top k sentences
        for sim, index in sorted_n_grams:
            if index in done_set:
                continue
            for x in range(index-self.n_gram_size+1, index+self.n_gram_size):
                done_set.add(x)

            n_gram_ids.append(index)
            k -= self.n_gram_size
            if k <= 0 or k < self.n_gram_size/2:
                break

        # generate summary
        return ".\n".join([self.n_grams[id][0] for id in n_gram_ids])