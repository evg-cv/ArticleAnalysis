import os

from utils.folder_file_manager import make_directory_if_not_exists


CUR_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'output'))
SENT_CLASSIFIER_MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'sent_classifier.pkl')
PERTINENT_MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'pertinent.pkl')
WORD_MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'pruned.word2vec.txt')

TRAINING_DATA_PATH = "/media/main/Data/Task/ArticleAnalysis/training_data.xlsx"
INPUT_EXCEL_PATH = "/media/main/Data/Task/ArticleAnalysis/input_file.xlsx"
