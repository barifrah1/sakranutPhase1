import numpy as np
import pandas as pd
from datetime import datetime

#case of failed projects  the other case is similar with fwe changes
data=pd.read_csv('ks-projects-201801.csv')
data['cat_sub_cat'] = data['main_category'] +'_'+data['category']
data['cat_sub_cat'] = data['main_category'] +'_'+data['category']
data =data.loc[s['state'].isin(['failed', 'successful'])] 
s=data[['cat_sub_cat','launched','deadline','backers','usd_pledged_real','state']]
# mean of successful
#data['backers_suc']=0
#data['pledged_suc']=0
#mean of failed
#data['backers_f']=0
#data['pledged_f']=0

#total_mean
#data['mean_pl']=0
#data['mean_bac']=0
for index, row in s.iterrows():
        d1=row['launched']
        csc=row['cat_sub_cat']
        val_ps=s.loc[(s['state']=='successful') & (s['deadline'] < d1) & (s['cat_sub_cat']==csc), 'usd_pledged_real'].mean()
        val_bs=s.loc[(s['state']=='successful') & (s['deadline'] < d1) & (s['cat_sub_cat']==csc), 'backers'].mean()
        #print(val)
        if pd.isnull(val_ps):
            continue
        data.loc[index,'pledged_suc']=int(val_ps)

        if pd.isnull(val_bs):
            continue
        data.loc[index,'backers_suc']=int(val_bs)

        val_pf=data.loc[(s['state']=='failed') & (s['deadline'] < d1) & (s['cat_sub_cat']==csc), 'usd_pledged_real'].mean()
        val_bf=data.loc[(s['state']=='failed') & (s['deadline'] < d1) & (s['cat_sub_cat']==csc), 'backers'].mean()
        if pd.isnull(val_pf):
            continue
        data.loc[index,'pledged_f']=int(val_pf)
        if pd.isnull(val_bf):
            continue
        data.loc[index,'backers_f']=int(val_bf)
        if (index%10000==0):
            print("index",index)

        
data.to_csv("outfx12.csv", index = False)