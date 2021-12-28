import sys
import pandas as pd
import numpy as np
import re
import pickle
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    INPUT
    data_filepath - a string representing file path from where we can load data
    
    OUTPUT
    X, y - python pandas series or dataframe representing X and y used for modeling
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('DisasterResponse', engine)
    X = df.message.values
    y = df[df.columns[4:]]
    category_names = y.columns.values
    
    return X, y, category_names


def tokenize(text):
    '''
    INPUT
    text - a string to be tokenized and cleaned for model purpose
    
    OUTPUT
    clean_tokens - a list containing cleaned words and elements in the string to be used for modeling
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT
    No input
    
    OUTPUT
    cv - a python scikit learn model after pipeline and grid search 
    '''
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        #'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'text_pipeline__vect__max_features': (None, 5000, 10000),
        #'text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_estimators': [10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs= -1)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    INPUT
    model - ML pipeline for predicting multiple target variables
    X_text - a pandas series representing existed test X data
    y_test - a pandas series representing numbers predicted by the model
    category_names - column names of target variable test set
    
    OUTPUT
    No output
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    INPUT 
    model - ML pipeline for predicting multiple target variables
    model_filepath -  filepath to save model
    
    OUTPUT
    No output
    """
    pickle.dump(model, open('classifier.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()