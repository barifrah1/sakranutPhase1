import numpy as np
import time
import pandas as pd
data=pd.read_csv('ks-projects-201801.csv')
data['cat_sub_cat'] = data['main_category'] +'_'+data['category']
data =data.loc[data['state'].isin(['failed', 'successful'])] 

start = time.time()
print(data[(data['state']=='successful') & (data['state']=='successful')]['usd_pledged_real'].mean())
end = time.time()
print("one way",end - start)

start = time.time()
x=data.loc[(data['state']=='successful') & (data['backers']>10), 'usd_pledged_real'].mean()
print(x)
end = time.time()
print("two way",end - start)