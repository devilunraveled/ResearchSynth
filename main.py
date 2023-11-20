from MatchSum import GetSummary as MatchSumSummary
from SumItUp import GetSummary as SumItUpSummary
from Pegasus import GetSummary as PEGASUSSummary

from parser.parser import *

SUMMARIES = {}

def generateAllSummaries( researchPaper ):
    try :
        SUMMARIES["MatchSum"] = MatchSumSummary(researchPaper)
        SUMMARIES["SumItUp"] = SumItUpSummary(researchPaper)
        SUMMARIES["PEGASUS"] = PEGASUSSummary(researchPaper)
    except Exception as E :
        print("Failed to generate summary")
        print(E)
        exit(0)

def getRawText( filePath ):
    try :
        researchPaper = Parser(filePath)
        if ( researchPaper is None or len(researchPaper) == 0 ):
            raise Exception(f"No text found for paper at {filePath}")
        return researchPaper
    except:
        print("Failed to parse ", filePath)
        exit(0)

def getInput():
    print("PATH must be wither ABSOLUTE or relative from the current working directory.")
    filePath = input("Enter file path:")
    return filePath

if __name__ == "__main__":
    filePath = getInput()
    researchPaper = getRawText(filePath)
    generateAllSummaries(researchPaper)

    for key, summary in SUMMARIES:
        print(f"Summary given by {key}:")
        print(summary)
