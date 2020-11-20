import numpy as np
import pandas as pd
from datetime import datetime

#case of failed projects  the other case is similar with fwe changes
data=pd.read_csv('ks-projects-201801.csv')
data = data.dropna()
data['cat_sub_cat'] = data['main_category'] +'_'+data['category']
s=data[data['state']=='failed']
s=s[['cat_sub_cat','launched','deadline','backers','usd_pledged_real']]
data['backers_s']=0
data['pledged_s']=0
for index, row in s.iterrows():
    print(index)
    if index>10:
        break
    d1=row['launched']
    csc=row['cat_sub_cat']
    val=s[(s['deadline'] < d1) & (s['cat_sub_cat']==csc)]['backers'].mean()
    #print(val)
    if pd.isnull(val):
        continue
    data.loc[index,'backers_s']=int(val)
    val1=s[(s['deadline'] < d1) & (s['cat_sub_cat']==csc)]['usd_pledged_real'].mean()
    print(val1)
    if pd.isnull(val1):
        continue
    data.loc[index,'pledged_s']=int(val1)
    #if (index%10000==0):
    #    print("index",index)
        
#data.to_csv("outf.csv", index = False)