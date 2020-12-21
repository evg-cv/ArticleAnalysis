import pandas as pd
import joblib
import os
import numpy as np

from operator import add
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from src.feature.extractor import GFeatureExtractor
from settings import TRAINING_DATA_PATH, PERTINENT_MODEL_PATH, SENT_CLASSIFIER_MODEL_PATH


class ClassifierTrainer:
    def __init__(self):
        self.feature_extractor = GFeatureExtractor()
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.training_data_frame = pd.read_excel(TRAINING_DATA_PATH)
        self.model_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                            "Naive Bayes", "QDA"]
        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=2, degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,
                decision_function_shape='ovr', random_state=None),
            SVC(gamma=2, C=1, probability=True),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

    @staticmethod
    def convert_str_array(array_str):
        list_str = array_str.replace("  ", " ").replace(" ", ",")
        last_comma = list_str.rfind(",")
        f_list_str = list_str[:last_comma] + list_str[last_comma + 1:]
        converted_array = np.array(literal_eval(f_list_str))

        return converted_array

    def train_best_model(self, model_path, x_data, y_data):
        x_train, x_test, y_train, y_test = \
            train_test_split(x_data, y_data, test_size=.3, random_state=42)

        scores = []
        for name, clf in zip(self.model_names, self.classifiers):
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            scores.append(score)
            # print(f"[INFO] model:{name}, score:{score}")

        # print(f"[INFO] The best model: {self.model_names[scores.index(max(scores))]}, {max(scores)}")
        best_clf = self.classifiers[scores.index(max(scores))]
        best_clf.fit(x_data, y_data)
        score = best_clf.score(x_test, y_test)
        print(f"[INFO] The accuracy of the best model: {self.model_names[scores.index(max(scores))]}, {score}")
        joblib.dump(best_clf, model_path)
        print(f"[INFO] Successfully saved in {model_path}")

        return

    def run(self):
        training_data = {}
        pertinent_x_data = []
        pertinent_y_data = []
        sent_x_data = []
        sent_y_data = []
        titles = self.training_data_frame["ARTICLE TITLE"].values.tolist()
        article_sentences = self.training_data_frame["ARTICLE SENTENCES"].values.tolist()
        labels = self.training_data_frame["LABEL"].values.tolist()
        title_indices = []
        for i, title in enumerate(titles):
            if type(title) is str:
                training_data[title] = []
                title_indices.append(i)

        for i, title_key in enumerate(list(training_data.keys())):
            if i == len(list(training_data.keys())) - 1:
                init_sentences = article_sentences[title_indices[i]:len(article_sentences)]
                init_labels = labels[title_indices[i]:len(labels)]
            else:
                init_sentences = article_sentences[title_indices[i]:title_indices[i + 1]]
                init_labels = labels[title_indices[i]:title_indices[i + 1]]
            for i_sent, i_label in zip(init_sentences, init_labels):
                if type(i_sent) is str:
                    if type(i_label) is str:
                        training_data[title_key].append([i_sent, i_label])
                    else:
                        training_data[title_key].append([i_sent, ""])

        for title_key in training_data.keys():
            title_feature = self.feature_extractor.get_feature_token_words(text=title_key)
            for sentence, label in training_data[title_key]:
                sentence_feature = self.feature_extractor.get_feature_token_words(text=sentence)
                pertinent_x_data.append(list(map(add, title_feature, sentence_feature)))
                if label != "":
                    sent_x_data.append(sentence_feature)
                    sent_y_data.append(label)
                    pertinent_y_data.append("Pertinent")
                else:
                    pertinent_y_data.append("Non-Pertinent")

        self.train_best_model(model_path=PERTINENT_MODEL_PATH, x_data=pertinent_x_data, y_data=pertinent_y_data)
        self.train_best_model(model_path=SENT_CLASSIFIER_MODEL_PATH, x_data=sent_x_data, y_data=sent_y_data)

        return


if __name__ == '__main__':
    ClassifierTrainer().run()
