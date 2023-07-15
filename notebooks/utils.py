import pandas as pd
import os
from typing import Optional
import re
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import unidecode

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from skops.io import dump, load

import pickle

random_state = 42


def get_label() -> str:
    return 'Survived'


def get_categorical_columns() -> list:
    return ['Pclass', 'Cabin', 'Sex', 'Embarked']


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = '../data/'
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return train_df, test_df


def make_balanced_dataframe(df: pd.DataFrame):
    label = get_label()
    df = df.copy()
    # make the set balanced
    not_survived = df[df[label] == 0]
    survived = df[df[label] == 1]

    if len(not_survived) > len(survived):
        not_survived = not_survived.sample(
            n=len(survived), random_state=random_state)
    elif len(not_survived) < len(survived):
        survived = survived.sample(
            n=len(not_survived), random_state=random_state)
    # shuffle the dataset
    df_cp = pd.concat([survived, not_survived]).sample(
        frac=1, random_state=random_state)
    return df_cp


def preprocess_dataframe(
        dataframe: pd.DataFrame,
        input_features: list,
        scaler=StandardScaler(),
        drop_na: bool = True,
        fill_na: bool = True,
        enable_categorical: bool = True,
        encoder=LabelEncoder(),
        test_split: Optional[float] = 0.2) -> pd.DataFrame:
    df = dataframe.copy()
    features = input_features.copy()
    categorical_cols = get_categorical_columns()

    # we know feature "Cabin" is 77% na values from explore.ipynb, so we remove that feature
    df = df.drop(columns=['Cabin'])
    if 'Cabin' in features:
        features.remove('Cabin')

    # ticket column is useless
    df = df.drop(columns=['Ticket'])
    if 'Ticket' in features:
        features.remove('Ticket')

    # these are bool types
    df['Has_Sibsp'] = (df['SibSp'] > 0).astype(int)
    features.append('Has_Sibsp')
    if 'Sibling' in features:
        features.remove('Sibling')
    df['Has_Parch'] = (df['Parch'] > 0).astype(int)
    features.append('Has_Parch')
    if 'Parch' in features:
        features.remove('Parch')

    for col in categorical_cols:
        if col in df.columns:
            # create categorical features
            if enable_categorical:
                print(f'Converting {col} to categorical')
                df[col] = df[col].astype('category')
            elif encoder is not None:
                print(f'Converting {col} to label')
                df[col] = encoder.fit_transform(
                    df[col].to_numpy().reshape(-1, 1))

    # drop rows with na values.
    if drop_na:
        for col in df.columns:
            na_idxs = pd.isna(df[col])
            if na_idxs.any():
                df = df[~na_idxs]
    elif fill_na:
        # pick a random value for each column
        for col in df.columns:
            na_idxs = pd.isna(df[col])
            if na_idxs.any():
                random_values = df[col][~na_idxs].sample(na_idxs.sum())
                df.loc[na_idxs, col] = random_values.values

    if test_split is not None:
        X_train, X_val, y_train,  y_val = train_test_split(
            df[features],
            df[get_label()],
            test_size=test_split,
            random_state=random_state)

        if scaler is not None:
            cols_to_scale = ['Age', 'Fare']
            X_train = scale_columns(X_train, cols_to_scale, scaler)
            X_val = scale_columns(X_val, cols_to_scale, scaler)
    elif scaler is not None:
        cols_to_scale = ['Age', 'Fare']
        return scale_columns(df[features], cols_to_scale, scaler)
    else:
        return df[features]
    return X_train, y_train, X_val, y_val


def scale_columns(df, cols, scaler):
    df = df.copy()
    for col in cols:
        df[col] = scaler.fit_transform(df[col].to_numpy().reshape(-1, 1))
    return df


def save_model(model, model_name: str):
    model_dir = '../models/'
    with open(os.path.join(model_dir, f'{model_name}.pkl'), 'wb') as f:
        pickle.dump(model, f)


def load_model(model_name: str):
    model_dir = '../models/'
    with open(os.path.join(model_dir, f'{model_name}.pkl'), 'rb') as f:
        return pickle.load(f)


def save_scikit_model(model, model_name: str):
    model_dir = '../models/'
    dump(model, os.path.join(model_dir, f'{model_name}.skops'))


def load_scikit_model(model_name: str):
    model_dir = '../models/'
    return load(os.path.join(model_dir, f'{model_name}.skops'))


class TextPreprocessor:
    # list of word types (nouns and adjectives) to leave in the text
    # , 'RB', 'RBS', 'RBR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    def_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']

    def __init__(self, apply_stemming: bool = True):
        self.apply_stemming = apply_stemming
        self.wnl = WordNetLemmatizer()
        self.stemmer = nltk.stem.SnowballStemmer(
            'english') if self.apply_stemming else None
        self.stop_words = stopwords.words('english')
        # remove mr and mrs since we have that information in different columns
        self.stop_words.append('mr', 'mrs')

    # functions to determine the type of a word
    @staticmethod
    def is_tag_noun(tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']

    @staticmethod
    def is_tag_verb(tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    @staticmethod
    def is_tag_adverb(tag):
        return tag in ['RB', 'RBR', 'RBS']

    @staticmethod
    def is_tag_adjective(tag):
        return tag in ['JJ', 'JJR', 'JJS']

    # transform tag forms
    def penn_to_wn(self, tag):
        if self.is_tag_adjective(tag):
            return nltk.stem.wordnet.wordnet.ADJ
        elif self.is_tag_noun(tag):
            return nltk.stem.wordnet.wordnet.NOUN
        elif self.is_tag_adverb(tag):
            return nltk.stem.wordnet.wordnet.ADV
        elif self.is_tag_verb(tag):
            return nltk.stem.wordnet.wordnet.VERB
        return nltk.stem.wordnet.wordnet.NOUN

    @staticmethod
    def unidecode_text(text):
        try:
            # pdb.set_trace()
            text = unidecode.unidecode(text)
        except:
            pass
        return text

    def pre_tokenization(self, doc: str):
        doc = doc.lower()
        doc = self.unidecode_text(doc)
        return doc

    def tokenize(self, doc):
        # prepare for tokenization
        doc = self.pre_tokenization(doc)
        # pattern for numbers | words of length=2 | punctuations | words of length=1
        pattern = re.compile(
            r'[0-9]+|\b[\w]{2,2}\b|[%.,_`!"&?\')({~@;:#}+-]+|\b[\w]{1,1}\b')
        # tokenize document
        doc_tok = word_tokenize(doc)
        # filter out patterns from words
        doc_tok = [x for x in doc_tok if x not in self.stop_words]
        doc_tok = [pattern.sub('', x) for x in doc_tok]
        # get rid of anything with length=1
        doc_tok = [x for x in doc_tok if len(x) > 1]
        # position tagging
        doc_tagged = nltk.pos_tag(doc_tok)
        # selecting nouns and adjectives
        doc_tagged = [(t[0], t[1])
                      for t in doc_tagged if t[1] in self.def_tags]
        # preparing lemmatization
        doc = [(t[0], self.penn_to_wn(t[1])) for t in doc_tagged]
        # lemmatization
        doc = [self.wnl.lemmatize(t[0], t[1]) for t in doc]
        # uncomment if you want stemming as well
        if self.apply_stemming:
            doc = [self.stemmer.stem(x) for x in doc]
        return doc
