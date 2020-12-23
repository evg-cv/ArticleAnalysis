import os
import ntpath
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from src.feature.extractor import GFeatureExtractor
from settings import TITLE_SIMILARITY_EXCEL_PATH, OUTPUT_DIR


class TitleSimilarity:
    def __init__(self):
        self.feature_extractor = GFeatureExtractor()

    def calculate_similarity_between_titles(self):
        file_name = ntpath.basename(TITLE_SIMILARITY_EXCEL_PATH).replace(".xlsx", "")
        output_file_path = os.path.join(OUTPUT_DIR, f"{file_name}_result.csv")
        data_frame = pd.read_excel(TITLE_SIMILARITY_EXCEL_PATH)
        origin_titles = data_frame["ORIGIN TITLE"].values.tolist()
        other_titles = data_frame["OTHER TITLES"].values.tolist()

        origin_output = []
        other_output = []
        similarity_output = []
        for origin_title, other_title in zip(origin_titles, other_titles):
            origin_output.append(origin_title)
            other_sub_titles = other_title.split(";")
            origin_feature = self.feature_extractor.get_feature_token_words(text=origin_title)
            for s_title in other_sub_titles:
                sub_title_feature = self.feature_extractor.get_feature_token_words(text=s_title)
                similarity = cosine_similarity([origin_feature], [sub_title_feature])
                other_output.append(s_title)
                origin_output.append("")
                similarity_output.append(similarity[0][0])
            origin_output = origin_output[:-1]

        pd.DataFrame([origin_output, other_output, similarity_output]).T.to_csv(output_file_path, index=True,
                                                                                header=["ORIGIN TITLE", "OTHER TITLE",
                                                                                        "SIMILARITY"], mode="w")

        print(f"[INFO] Successfully saved into {output_file_path}")

        return


if __name__ == '__main__':
    TitleSimilarity().calculate_similarity_between_titles()
