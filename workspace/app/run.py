import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load dataDisasterResponse
engine = create_engine('sqlite:////home/workspace/models/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)




# load model
model = joblib.load("/home/workspace/models/classifier.pickle")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    Y=pd.DataFrame(Y).astype('int')
    M=pd.DataFrame(np.random.rand(1,36),columns=Y.columns)   
    for i in Y.columns:
        M[i]=Y[i].sum()
    #取TOP10
    df1=M
    df2 = pd.DataFrame(df1.values.T, index=df1.columns, columns=df1.index)#转置
    df3=df2.sort_values(axis = 0,ascending = False,by=0)
    df4=df3.head(10)
    df5=pd.DataFrame(df4.values.T,index=df4.columns,columns=df4.index)
    
    #画柱状图
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import figsize
    x=np.arange(len(df5.columns))
    A=df5.values.tolist()
    y=A[0]
    figsize(10, 10)
    plt.bar(x,y,color='rgb',tick_label=df5.columns)
    ''' 添加数据标签'''
    for x,y in zip(x,y):
        plt.text(x,y ,'%.2f'%y ,ha='center',va='bottom')
    '''显示'''
    plt.xticks(rotation=45)
    plt.title('Count of Message in Categories(TOP 10)')#绘制标题
    plt.savefig('./Count of Message in Categories(TOP 10)')#保存图片
    plt.show()
    
    
    #画饼状图
    import matplotlib.pyplot as plt
    O=df5.values.tolist()
    J=O[0]
    plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
    plt.figure(figsize=(5,5))#将画布设定为正方形，则绘制的饼图是正圆
    label=df5.columns#定义饼图的标签，标签是列表
    explode=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]#设定各项距离圆心n个半径
    #plt.pie(values[-1,3:6],explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
    values=J
    plt.pie(values,explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
    plt.title('Percentages of Message in Categories(TOP 10)')#绘制标题
    plt.savefig('./Percentages of Message in Categories(TOP 10)')#保存图片
    plt.show()
    
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    


    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()