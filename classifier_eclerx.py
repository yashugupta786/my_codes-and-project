import pandas as pd
import labels_dict
from sklearn.utils import resample
import string
import re
import nltk
import numpy as np
import itertools
from sklearn.metrics import classification_report
lemma = nltk.WordNetLemmatizer()
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
df=pd.read_excel("training_data.xlsx")
df.dropna(inplace=True)
df["Suggested Theme"]=df["Suggested Theme"].map(labels_dict.labels)
print(df["Suggested Theme"].value_counts())
exit()
# print(df["Suggested Theme"].value_counts())
import matplotlib.pyplot as plt
class Comments_suggestions(object):
    def __init__(self,df):
        self.my_dataframe = df

        df_majority = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 5]  # 1== promoters
        df_minority1 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 1]  # 0=Detractor
        df_minority2 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 2]
        df_minority3 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 3]
        df_minority4 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 4]
        df_minority5 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 6]
        df_minority6 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 7]
        df_minority7 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 8]
        df_minority8 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 9]
        df_minority9 = self.my_dataframe[self.my_dataframe["Suggested Theme"] == 10]

        df_minority_upsampled1 = resample(df_minority1,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)
        df_minority_upsampled2 = resample(df_minority2,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)
        df_minority_upsampled3 = resample(df_minority3,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)
        df_minority_upsampled4 = resample(df_minority4,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)
        df_minority_upsampled5 = resample(df_minority5,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)
        df_minority_upsampled6 = resample(df_minority6,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)
        df_minority_upsampled7= resample(df_minority7,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)
        df_minority_upsampled8 = resample(df_minority8,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)
        df_minority_upsampled9 = resample(df_minority9,
                                          replace=True,  # sample with replacement
                                          n_samples=int(215),  # to match majority class
                                          random_state=123)

        new_df = pd.concat([df_majority, df_minority_upsampled1, df_minority_upsampled2,
                            df_minority_upsampled3,df_minority_upsampled4,df_minority_upsampled5,
                            df_minority_upsampled6,df_minority_upsampled7,df_minority_upsampled8,df_minority_upsampled9])
        # print(new_df["Suggested Theme"].value_counts())
        new_df.dropna(inplace=True)
        self.feature_questions = new_df["comments and improvemnet"].tolist()
        self.labels = new_df["Suggested Theme"].tolist()

    def plot_confusion_matrix(self,cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(4,4))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()

    # Plot normalized confusion matrix

    def comments_train(self):
        try:
            _feature_questions = []
            _feature_questions_noun = []
            _feature_questions_verb = []
            feature_questions_punct = []
            global TEXT_CLASSIFIER
            feature_lables = self.labels
            feature_questions_lwr = [x.lower() for x in self.feature_questions]
            for i in feature_questions_lwr:
                translation_table = dict.fromkeys(map(ord, string.punctuation), ' ')
                string2 = i.translate(translation_table)  # translating string1
                feature_questions_punct.append(string2)
            feature_questions_spc = [s for s in feature_questions_punct if s]
            for i in feature_questions_spc:
                _feature_questions.append(re.sub('\s+', ' ', i).strip())
            print("lemma start")
            feature_questions_noun = [[lemma.lemmatize(word, 'n') for word in sentence.split(" ")] for sentence in
                                      _feature_questions]
            for i in feature_questions_noun:
                _feature_questions_noun.append(" ".join(i))
            feature_questions_verb = [[lemma.lemmatize(word, 'v') for word in sentence.split(" ")] for sentence in
                                      _feature_questions_noun]
            for i in feature_questions_verb:
                _feature_questions_verb.append(" ".join(i))
            print("lemma finish")
            interpretation_all_data = feature_lables
            print(_feature_questions_verb)
            print(interpretation_all_data)
            X_train, X_test, y_train, y_test = train_test_split(_feature_questions_verb, interpretation_all_data,
                                                                random_state=42,test_size=0.1)
            print("TFIDF vectorization start")
            tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            text_classifier = Pipeline([
                ('vectorizer', tfidf_vectorizer),
                ('clf', SGDClassifier(loss='log',
                                      n_jobs=-1,
                                      max_iter=15,
                                      random_state=0,
                                      shuffle=True,
                                      tol=0.01))
            ])
            print("model training start -----------------")
            TEXT_CLASSIFIER = text_classifier.fit(X_train, y_train)
            pred = TEXT_CLASSIFIER.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            print(accuracy)
            F1score = f1_score(y_test, pred, average="macro")

            #working code

            # cm = confusion_matrix(y_test, pred)
            # self.plot_confusion_matrix(cm,
            #                       normalize=True,
            #                       target_names=["1","2","3","4","5","6","7","8","9","10"],
            #                       title="Confusion Matrix - Theme classification")



            # plt.figure(figsize=(5.5, 4))
            # cm_df = pd.DataFrame(cm)
            # sns.heatmap(cm_df, annot=True)
            # # plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
            # plt.ylabel('True label')
            # plt.xlabel('Predicted label')
            # plt.show()
            target_names=['Product Expectation',
'Price/Value/Contract','User Experience ','General Positive',
'Reliability/Connectivity','Other LOB Feedback','Does Not Use','Support','Incorrect Installation','Other']
            classreport = classification_report(y_test, pred,target_names=target_names)

            # print(F1score)
            # testdf=pd.DataFrame(X_test,columns=["comments and improvements"])
            # testdf["original_prediction"]=y_test
            # testdf["prediction"]=pred
            # testdf["prediction"]=testdf["prediction"].map(labels_dict.label_to_original)
            # testdf["original_prediction"]=testdf["original_prediction"].map(labels_dict.label_to_original)
            # testdf.to_csv("final_predictions_comment.csv")
            global flag
            flag = True
        except Exception as exception:
            print(str(exception))


obj1=Comments_suggestions(df)
obj1.comments_train()