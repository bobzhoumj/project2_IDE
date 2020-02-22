
import nltk

#增加nltk的权限
import ssl 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#nltk.download()




import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
import os


import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import pickle 
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn import datasets

#分词加载
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


#加载数据
def load_data(database_filepath):
    
    #engine = create_engine('sqlite:///C://Users//ThinkPad//Desktop//project2//workspace//DisasterResponse.db')
#     engine = create_engine(database_filepath)


    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='DisasterResponse',con=engine,index_col='id')
    
#     df=pd.read_sql_query('select * from DisasterResponse', engine)
    category_names=['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    X = df['message']
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
   
    
    return X,Y,category_names


def tokenize(text):
    text = re.sub(r"[^z-zA-Z0-9]"," " ,text.lower())
    
    tokens = word_tokenize(text)
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


#建管道，建立多分类模型：决策树
def build_model():
    
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier( DecisionTreeClassifier(random_state =42), n_jobs = -1))
         ])
    return pipeline
    
    


#评估模型，评价维度：F1值，精确度，召回率
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred=model.predict(X_test)
    
    Y_pred=pd.DataFrame(Y_pred,columns=['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']).astype('int')
    Y_test=pd.DataFrame(Y_test).astype('int')
    for i in Y_test.columns:
        print(i)
        print('f1_scroe:',f1_score(Y_test[i], Y_pred[i], average='macro'))
        print('precision_score:',precision_score(Y_test[i], Y_pred[i], average="macro"))
        print('recall_score:',recall_score(Y_test[i], Y_pred[i], average="macro"),'\n')
    

    pass




#保存模型
def save_model(model, model_filepath):
    
#1.保存成Python支持的文件格式Pickle
#在当前目录下可以看到svm.pickle
#     with open('classifier.pickle','wb') as fw:
    with open(model_filepath,'wb') as fw:  
        pickle.dump(model,fw)
#加载svm.pickle
#     with open('classifier.pickle','rb') as fr:
    with open(model_filepath,'rb') as fr:
        new_pipeline = pickle.load(fr)
  


#主函数
def main():
    if len(os.getcwd()) != 0:

#         model_filepath = os.getcwd()
#         database_filepath = os.getcwd()
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
#         a='sqlite:///'
#         database_filepath=a+'/home/workspace/data/DisasterResponse.db'
    
        
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        #网格搜索来优化模型参数
        from sklearn.model_selection import GridSearchCV
        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000),
            'tfidf__use_idf': (True, False)}
        cv = GridSearchCV(model, param_grid=parameters)     
        
        fit = cv.fit(X_train,Y_train) #GridSearchCV模型拟合训练集数据，并返回训练器集合为fit
        print("\nBest Parameters:", fit.best_params_)
        Y_pred=fit.best_estimator_.predict(X_test)
        Y_pred=pd.DataFrame(Y_pred,columns=['related', 'request', 'offer', 'aid_related', 'medical_help',
                                 'medical_products', 'search_and_rescue', 'security', 'military',
                                 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
                                 'missing_people', 'refugees', 'death', 'other_aid',
                                 'infrastructure_related', 'transport', 'buildings', 'electricity',
                                 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                                 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                                 'other_weather', 'direct_report']).astype('int')
        Y_test=pd.DataFrame(Y_test).astype('int')
        model=fit.best_estimator_
               
        
        print('optimize Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

#运行主函数
if __name__ == '__main__':
    main()