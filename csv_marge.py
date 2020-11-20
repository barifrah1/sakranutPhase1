import pandas as pd 

data=pd.read_csv('ks-projects-201801.csv')
out=pd.read_csv('outf.csv')
data = data.dropna()
out = out.dropna()
data.loc[(data['state']=='failed'), 'backers_s'] = out.loc[(out['state']=='failed'), 'backers_s']
data.loc[(data['state']=='failed'), 'pledged_s'] = out.loc[(out['state']=='failed'), 'pledged_s']



data.to_csv("ks-projects-2018011.csv", index = False)