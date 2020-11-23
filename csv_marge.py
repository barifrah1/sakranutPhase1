import pandas as pd 

data=pd.read_csv('ks-projects-201801.csv')
out=pd.read_csv('outfx.csv')
data = data.dropna()
out = out.dropna()
out.loc[(out['Unnamed']<=40000), 'pledged_suc'] = data.loc[(data['Unnamed']<=40000), 'pledged_suc']
out.loc[(out['Unnamed']<=40000), 'backers_suc'] = data.loc[(data['Unnamed']<=40000), 'backers_suc']
out.loc[(out['Unnamed']<=40000), 'pledged_f'] = data.loc[(data['Unnamed']<=40000), 'pledged_f']
out.loc[(out['Unnamed']<=40000), 'backers_f'] = data.loc[(data['Unnamed']<=40000), 'backers_f']



out.to_csv("ks-projects-2018011.csv", index = False)