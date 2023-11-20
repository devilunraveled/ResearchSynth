import matplotlib.pyplot as Plot

modelList = ["SumItUp", "MatchSum", "PEGASUSLarge", "PEGASUSPretrained_10000", "PEGASUSPretrained_80000", "PEGASUSARCHIVE", "BigBirdPEGASUS"]
timeTakenPerSummary = [0.3 , 18.2, 26.3, 27.5, 27.2, 23.5, 285]
rougeF1Score =        [26.6, 36.9, 32.4, 34.1, 38.1, 39.2, 42.3]

#Create a scatter Plot
Plot.scatter( rougeF1Score, timeTakenPerSummary)
Plot.ylabel("Time Taken")
Plot.xlabel("Rouge F1 Score")
Plot.show()
