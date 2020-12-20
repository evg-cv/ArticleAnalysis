import os
import re
import joblib
import ntpath
import pandas as pd
import nltk

from nltk.tokenize import sent_tokenize
from src.feature.extractor import GFeatureExtractor
from src.preprocess.tokenizer import TextPreprocessor
from settings import SENT_CLASSIFIER_MODEL_PATH, PERTINENT_MODEL_PATH, OUTPUT_DIR


class ArticleAnalyzer:
    def __init__(self):
        self.feature_extractor = GFeatureExtractor()
        self.text_processor = TextPreprocessor()
        self.pertinent_model = joblib.load(PERTINENT_MODEL_PATH)
        self.sent_model = joblib.load(SENT_CLASSIFIER_MODEL_PATH)

    def analyze_article(self, title, text):
        sent_categories = []
        pertinents = []
        text = text.replace("“", "{").replace("”", "{").replace("'", "")
        title_feature = self.feature_extractor.get_feature_token_words(text=title)
        quote_st_indices = re.finditer("{", text)
        for i, q_index in enumerate(quote_st_indices):
            if i % 2 == 1:
                continue
            i_index = q_index.start() + 1
            while text[i_index] != "{":
                if (text[i_index] == "." or text[i_index] == "!" or text[i_index] == "?") and text[i_index + 1] != "{":
                    text = text[:i_index] + "," + text[i_index + 1:]
                i_index += 1
                if i_index >= len(text):
                    break
        text = text.replace("{", "")
        sentences = sent_tokenize(text=text)
        for sent in sentences:
            sent_feature = self.feature_extractor.get_feature_token_words(text=sent)
            input_feature = [title_feature + sent_feature]
            pertinent_ret = self.pertinent_model.predict(input_feature)
            if pertinent_ret == "Pertinent":
                pertinents.append("True")
                sent_category = self.sent_model.predict(input_feature)
                sent_categories.append(sent_category[0])
            else:
                pertinents.append("False")
                sent_categories.append("")

        return sentences, sent_categories, pertinents

    def run(self, file_path):
        file_name = ntpath.basename(file_path).replace(".xlsx", "")
        output_file_path = os.path.join(OUTPUT_DIR, f"{file_name}_result.csv")
        data_frame = pd.read_excel(file_path)
        article_titles = data_frame["ARTICLE TITLE"].values.tolist()
        article_contents = data_frame["ARTICLE CONTENT"].values.tolist()
        for art_title, art_content in zip(article_titles, article_contents):
            sentences, sent_categories, pertinents = self.analyze_article(title=art_title, text=art_content)
            pd.DataFrame([[art_title], sentences, sent_categories, pertinents]).T.to_csv(output_file_path, index=True,
                                                                                         header=["ARTICLE TITLE",
                                                                                                 "ARTICLE SENTENCES",
                                                                                                 "CATEGORY",
                                                                                                 "PERTINENT"],
                                                                                         mode="a")

        print(f"[INFO] Successfully saved the result in {output_file_path}")

        return


if __name__ == '__main__':
    ArticleAnalyzer().analyze_article(title="",
                                      text='')
