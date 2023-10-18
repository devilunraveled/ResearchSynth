from parser import Parser
import os

for fileName in os.listdir('../papers'):
    Parser(os.path.join('../papers', fileName))
    print(fileName, "parsed!")
