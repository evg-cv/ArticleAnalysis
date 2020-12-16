import os

from utils.folder_file_manager import make_directory_if_not_exists


CUR_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'output'))
WORD_MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'pruned.word2vec.txt')

INPUT_EXCEL_PATH = ""
