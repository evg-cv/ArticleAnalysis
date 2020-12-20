from src.article.analyzer import ArticleAnalyzer
from settings import INPUT_EXCEL_PATH


if __name__ == '__main__':
    ArticleAnalyzer().run(file_path=INPUT_EXCEL_PATH)
