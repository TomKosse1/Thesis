import csv
import glob
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics

def textFeature(mode):
    """ Creates dataframe with Labels and Text per review, File_ID as index
    Paramater mode: 'train' or 'test'
    Returns pandas dataframe"""
    
    classlist = ['negative', 'positive']
    data = pd.DataFrame()

    for label in classlist:
        path = 'C:\\Users\\Tom\\Documents\\Informatiekunde\\Thesis\\data\\' + mode + '\\' + label + '\\'
        allFiles = glob.glob(path + "*.txt")
        df1 = pd.DataFrame()
        for review in allFiles:
            title = review.strip('.txt').split('\\')[-1]
            text = open(review, 'r', encoding='utf8').read()
            df = pd.DataFrame({'File': [title], 'Text': [text], 'Label': [label]}).set_index('File')
            df1 = df1.append(df)
        data = data.append(df1)
    
    return data

def clfFeature(feature, mode):
    """Creates dataframe with certain Feature per review, file_ID as index
    Parameter feature: 'sentiment', 'actors', 'directors', 'genre' or 'titles'
    Parameter mode: 'train' or 'text'
    Returns pandas dataframe"""
    
    feature_path = 'C:\\Users\\Tom\\Documents\\Informatiekunde\\Thesis\\features\\' + feature + '.txt'
    classlist = ['negative', 'positive']
    features = pd.DataFrame()

    for label in classlist:
        path = 'C:\\Users\\Tom\\Documents\\Informatiekunde\\Thesis\\data\\' + mode + '\\' + label + '\\'
        allFiles = glob.glob(path + "*.txt")
        for review in allFiles:
            title = review.strip('.txt').split('\\')[-1]
            file = open(review, 'r', encoding='utf8').read().lower()
            wordlist = []
            featreader = csv.reader(open(feature_path, 'r'), delimiter= '\n')
            for word in featreader:
                if word[0] in file:
                    wordlist.append(word[0])
            df = pd.DataFrame({'File': [title], feature.capitalize(): [', '.join(wordlist)]}).set_index('File')
            features = features.append(df)
    
    return features

def createFeatureFrame(mode):
    """Combines the Text dataframe and the Feature dataframe
    Parameter mode: 'train' or 'test'
    Returns pandas dataframe"""
    
    text = textFeature(mode)
    sentiment = clfFeature('sentiment', mode)
    actors = clfFeature('actors', mode)
    directors = clfFeature('directors', mode)
    genre = clfFeature('genre', mode)
    titles = clfFeature('titles', mode)
    featureframe = pd.concat([text, sentiment, actors, directors, genre, titles], axis=1)
    
    return featureframe

def combineFeatures(featurelist):
    """Combines different feature sets
    Parameter featurelist: list containing two or more features to combine
    Possible items in featurelist: 'sentiment', 'actors', 'directors', 'genre' and 'titles'
    Returns pandas dataframe"""
    
    cap_list = []
    for item in featurelist:
        cap_list.append(item.capitalize())
    features['Features'] = features[cap_list].apply(lambda x: ', '.join(x), axis=1)
    
    return features

def performClassification(ngram, df, mode = None, split = 0.9):
    """Fits data to the model and performs classification for Naive Bayes and Linear SVC
    Parameter ngram: 1, 2 or 3
    Parameter df: the dataframe to be ditted to the model
        if combined features: created dataframe from combineFeatures
        if single features: created dataframe from createFeatureFrame, mode needs to be specified
    Parameter mode: 
        if combined features: default (None)
        if single feature: 'sentiment', 'actors', 'directors', 'genre' or 'titles'
    Parameter split: float (percentage of data to be used for training), default = 0.9
    Returns computed accuracy, precision, recall and F1 scores"""
    
    if type(mode) == str:
        X = df[mode.capitalize()]
    else:
        X = df.Features
        
    y = df.Label

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size = split)
    
    vect = CountVectorizer(analyzer='word', ngram_range=(ngram,ngram))
    
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    
    nb = MultinomialNB()
    svm = LinearSVC(random_state = 1)

    nb.fit(X_train_dtm, y_train)
    svm.fit(X_train_dtm, y_train)

    nb_pred_class = nb.predict(X_test_dtm)
    svm_pred_class = svm.predict(X_test_dtm)

    nb_accuracy = metrics.accuracy_score(y_test, nb_pred_class)
    nb_precision = metrics.precision_score(y_test, nb_pred_class, pos_label='negative')
    nb_recall = metrics.recall_score(y_test, nb_pred_class, pos_label='negative')
    nb_f1 = metrics.f1_score(y_test, nb_pred_class, pos_label='negative')

    svm_accuracy = metrics.accuracy_score(y_test, svm_pred_class)
    svm_precision = metrics.precision_score(y_test, svm_pred_class, pos_label='negative')
    svm_recall = metrics.recall_score(y_test, svm_pred_class, pos_label='negative')
    svm_f1 = metrics.f1_score(y_test, svm_pred_class, pos_label='negative')

    print('=====Naive Bayes===== \t =====Linear SVC=====' )
    print('Accuracy score \t\t Accuracy score')
    print(round((nb_accuracy * 100), 1), '\t\t\t', round((svm_accuracy * 100), 1), '\n')
    print('Precision \t\t Precision')
    print(round((nb_precision * 100), 1), '\t\t\t', round((svm_precision * 100), 1), '\n')
    print('Recall \t\t\t Recall')
    print(round((nb_recall * 100), 1), '\t\t\t', round((svm_recall * 100), 1), '\n')
    print('F1-score \t\t F1-score')
    print(round((nb_f1 * 100), 1), '\t\t\t', round((svm_f1 * 100), 1))