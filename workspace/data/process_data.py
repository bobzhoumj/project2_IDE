#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import os
import sqlite3
from sqlalchemy import create_engine


#下载数据messages,categories，并且依据id进行合并，合并后的数据命名为df,并返回
def load_data(messages_filepath, categories_filepath):
#     messages = pd.read_csv(os.path.join(messages_filepath,'disaster_messages.csv'),encoding='utf-8')

#     categories = pd.read_csv(os.path.join(messages_filepath,'disaster_categories.csv'),encoding='utf-8')
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df=pd.merge(messages,categories,on='id',how='inner')
    return df


#数据清洗，将字段categories中字段名称和值相应切片清洗出来，并且去除重复值
def clean_data(df):
    for i in range(0,36,1):
        df['a'+str(i)]=df['categories'].str.split(';').str.get(i)
    for i in range(0,36,1):
        df[df['a'+str(i)].str.split('-').str.get(0)[0]]=df['a'+str(i)].str.split('-').str.get(1)
    df.drop(['categories', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7',
       'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17',
       'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27',
       'a28', 'a29', 'a30', 'a31', 'a32', 'a33', 'a34', 'a35'],axis=1,inplace=True)
    isduplictae=None
    isduplictae=df.duplicated()
    isduplictae_num=None
    isduplictae_num=sum(df.duplicated()==True)
    if isduplictae_num>0:
        new_df=df.drop_duplicates()
        sum(new_df.duplicated()==True)

    return new_df

#使用create_engine将清洗后的数据储存在数据库中，库名DisasterResponse
def save_data(df, database_filename):
#     engine = create_engine('sqlite:///DisasterResponse.db')
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False) 


#主函数
def main():
    if len(os.getcwd()) != 0:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


        
#运行主函数       
if __name__ == '__main__':
    
    main()




    

