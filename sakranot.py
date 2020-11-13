import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('ks-projects-201801.csv')

def split_train_test(data):
    msk = np.random.rand(len(data)) < 0.8
    train = df[msk]
    test = df[~msk]

#get dataframe and return numpy arrey and columns names and range list
def preprocess(data):
    data['deadline']=data['deadline']=pd.to_datetime(data['deadline'], format="%Y/%m/%d").dt.date
    data['launched']=data['launched']=pd.to_datetime(data['launched'], format="%Y/%m/%d").dt.date
    data['duration'] = (data['deadline'] - data['launched']).dt.days
    #data['launch_year']=pd.to_datetime(data['launched'], format="%Y/%m/%d").dt.year
    #data['ratio']=data['usd_pledged_real']/data['usd_goal_real']
    #feature removal
    data = data.drop(['ID','goal','pledged','usd pledged','name','deadline','launched'],1)
    #ignore live\cancel\other because lower density
    data = data.loc[data['state'].isin(['failed','successful'])]
    # drop na values
    data=data.dropna()
    #data['country'].value_counts(normalize=True)
    #=> US and gb is almost 80% of the data =>dimension reduction
    data.loc[~data['country'].isin(['US','GB']),'country']='OTHER'
    cat_columns=['category','main_category','currency','state','country']
    # convert to categorial var
    data['category'] = data['category'].astype('category')
    data['main_category'] = data['main_category'].astype('category')
    data['currency'] = data['currency'].astype('category')
    data['state'] = data['state'].astype('category')
    data['country'] = data['country'].astype('category')
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    Var_Corr = data.corr()
    # plot the heatmap and annotation on it
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    #drop phase 2 according to correlation matrix result
    #remove backers due to high correlation with pladge
    data = data.drop(['currency','category','backers'],1)
    #data['usd_pledged_real'].value_counts(normalize=False) 
    data['goal_level']=0 # 0 - low ,1-mid,2 - high,3-very high 
    data['pledged_level']=0 # 0 - low ,1-mid,2 - high,3-very high 
    data['duration_level']=0 # 0 - short-mid , 1-mid-long , 
    # goal bins according distribution
    data.loc[(data['usd_goal_real']<=2000),'goal_level']=0
    data.loc[((data['usd_goal_real']>2000) & (data['usd_goal_real']<=5000)),'goal_level']=1
    data.loc[((data['usd_goal_real']>5000) & (data['usd_goal_real']<=15000)),'goal_level']=2
    data.loc[(data['usd_goal_real']>15000),'goal_level']=3
    # pledges bins
    data.loc[(data['usd_pledged_real']<=5),'pledged_level']=0
    data.loc[((data['usd_pledged_real']>5) & (data['usd_pledged_real']<=788)),'pledged_level']=1
    data.loc[((data['usd_pledged_real']>788) & (data['usd_pledged_real']<=4608)),'pledged_level']=2
    data.loc[(data['usd_pledged_real']>4608),'pledged_level']=3
    #duration bins
    data.loc[(data['duration']<=30),'duration_level']=0
    data.loc[(data['duration']>30),'duration_level']=1
    # phase 3 of drops results of bins creation
    data = data.drop(['usd_pledged_real','usd_goal_real','duration'],1)
    # reorder columns
    data=data[['main_category','country','goal_level','pledged_level','duration_level','state']]
    data_Arr=data.values
    columns_names=list(data.columns) 
    # list with tuple (column,unique value range)
    columns_names_with_unique_range=[]
    for x in columns_names:
        columns_names_with_unique_range.append((x,len(data[x].unique())))
    return data_Arr,columns_names_with_unique_range

# got numpy array and split to train and test with respect to ratio parm
def split_train_test(ndarr):
    split = np.random.rand(ndarr.shape[0]) < 0.8
    train=ndarr[split]
    test=ndarr[~split]
    return train,test
