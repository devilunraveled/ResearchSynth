import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import nltk
from nltk.tokenize import word_tokenize
import math
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def normalize_sentence_length(sLen):
    sLenmax = max(sLen)
    if sLenmax == 0:
        return [0.0] * len(sLen)  # Avoid division by zero
    return [sLeni / sLenmax for sLeni in sLen]

def normalize_tfidf(tfidf_scores):
    tfidf_max = max(tfidf_scores)
    if tfidf_max == 0:
        return [0.0] * len(tfidf_scores)  # Avoid division by zero
    return [(tfidf_i / tfidf_max) for tfidf_i in tfidf_scores]

def normalize_noun_verb_phrase(nvpi):
    nvpi_max = max(nvpi)
    if nvpi_max == 0:
        return [0.0] * len(nvpi)  # Avoid division by zero
    return [(nvpi_i / nvpi_max) for nvpi_i in nvpi]

def normalize_proper_noun(PNi):
    PNmax = max(PNi)
    if PNmax == 0:
        return [0.0] * len(PNi)  # Avoid division by zero
    return [(PNi_i / PNmax) for PNi_i in PNi]

def normalize_cosine_similarity(ACS):
    ACSmax = max(ACS)
    if ACSmax == 0:
        return [0.0] * len(ACS)  # Avoid division by zero
    return [(ACS_i / ACSmax) for ACS_i in ACS]

def normalize_cue_phrases(CPi, tCP):
    if tCP == 0:
        return [0.0] * len(CPi)  # Avoid division by zero
    return [(CPi_i / tCP) for CPi_i in CPi]

def GetSummary(article):
    sentences = article.split(" .")

    clean_article = []
    for sen in sentences:
      sen = sen.replace('\n', ' ')
      sen = ' '.join(word for word in sen.split() if not word.startswith('@'))
      sen = re.sub(r'\W+', ' ', sen)
      clean_article.append(sen)
    
    cue_phrases = ["the paper describes", "in conclusion", "in summary", "our investigation", "the best", "the most important", "in particular", "according to the study", "significantly", "important", "hardly", "impossible"]
    scores = []
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_article)
    
    for i, sentence in enumerate(clean_article):
        sLen = len(word_tokenize(sentence))
        sPos = i / len(sentences)

        pos_tags = nltk.pos_tag(word_tokenize(sentence))
        nvpi = len([word for word, pos in pos_tags if pos in ['NN', 'NNS', 'VB', 'VBD', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])
        PNi = len([word for word, pos in pos_tags if pos in ['NNP', 'NNPS']])

        current_tfidf = tfidf_matrix[i].toarray()
        acs = cosine_similarity(current_tfidf, tfidf_matrix).mean()
        cp_count = sum(1 for phrase in cue_phrases if phrase in sentence.lower())
        tf_idf = current_tfidf.sum()

        scores.append({
            "sentence": sentence,
            "F1": sLen,
            "F2": sPos,
            "F3": tf_idf,
            "F4": nvpi,
            "F5": PNi,               
            "F6": acs,
            "F7": cp_count,
        })
    
    normalized_scores = []
    df = pd.DataFrame(scores)
        
    df['F1'] = normalize_sentence_length(df['F1'])
    df['F3'] = normalize_tfidf(df['F3'])
    df['F4'] = normalize_noun_verb_phrase(df['F4'])
    df['F5'] = normalize_proper_noun(df['F5'])
    df['F6'] = normalize_cosine_similarity(df['F6'])
    df['F7'] = normalize_cue_phrases(df['F7'], df['F7'].sum())
        
    normalized_scores.append(df)

    df['total_score'] = df[['F1', 'F3', 'F4', 'F5', 'F6', 'F7']].sum(axis=1)

    num_sentences = 10
    similarity_threshold=0.8

    df_sorted = df.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(df_sorted['sentence'])

    summary_sen = []
    for _, row in df_sorted.iterrows():
        sen = row['sentence']
        if not summary_sen:
            summary_sen.append(sen)
        else:
            # Calculate cosine similarity with existing summary
            similarity = cosine_similarity(
                tfidf_vectorizer.transform([sen]),
                tfidf_vectorizer.transform(summary_sen)
            )
            # If the sentence is not too similar to the existing summary, add it
            if np.max(similarity) < similarity_threshold:
                summary_sen.append(sen)
            # Stop adding sentences if we've reached the desired length
        if len(summary_sen) == num_sentences:
            break

    summary = '. '.join(summary_sen)
    tool.correct(summary)
    return summary
