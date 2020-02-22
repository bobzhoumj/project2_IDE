# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    '''执行命令如下：
      cd /home/workspace/data/
      python process_data.py disaster_messages.csv disaster_categories.csv /home/workspace/models/DisasterResponse.db
    '''
    
    
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    '''执行命令如下：
    cd /home/workspace/models/
    python train_classifier.py DisasterResponse.db classifier.pickle
    '''

2. Run the following command in the app's directory to run your web app.
    #添加2个可视化图表
    cd /home/workspace/app/
    `python run.py`

3. Go to http://0.0.0.0:3001/
