from TextAnalyzer import TextAnalyser
from config import *


def main():
    analyzer = TextAnalyser(INPUT_XLSX,
                            OUTPUT_XLSX,
                            PERFECT_ANSWER_PATH,
                            STOPS_PATH,
                            SYMBOLS)
    analyzer.run()


if __name__ == '__main__':
    main()
