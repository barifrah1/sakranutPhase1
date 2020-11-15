import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args['fileName'], nrows=60000)

    def split_train_test(self):
        msk = np.random.rand(len(self.data)) < self.args['trainSize']
        train = self.data[msk]
        test = self.data[~msk]
        return train, test
    # get dataframe and return numpy arrey and columns names and range list

    def preprocess(self):
        self.data['deadline'] = self.data['deadline'] = pd.to_datetime(
            self.data['deadline'], format="%Y/%m/%d").dt.date
        self.data['launched'] = self.data['launched'] = pd.to_datetime(
            self.data['launched'], format="%Y/%m/%d").dt.date
        self.data['duration'] = (
            self.data['deadline'] - self.data['launched']).dt.days
        #self.data['launch_year']=pd.to_datetime(self.data['launched'], format="%Y/%m/%d").dt.year
        # self.data['ratio']=self.data['usd_pledged_real']/self.data['usd_goal_real']
        # feature removal
        self.data = self.data.drop(['ID', 'goal', 'pledged', 'usd pledged',
                                    'name', 'deadline', 'launched'], 1)
        # ignore live\cancel\other because lower density
        self.data = self.data.loc[self.data['state'].isin(
            ['failed', 'successful'])]
        # drop na values
        self.data = self.data.dropna()
        # self.data['country'].value_counts(normalize=True)
        # => US and gb is almost 80% of the self.data =>dimension reduction
        self.data.loc[~self.data['country'].isin(
            ['US', 'GB']), 'country'] = 'OTHER'
        cat_columns = ['category', 'main_category',
                       'currency', 'state', 'country']
        # convert to categorial var
        self.data['category'] = self.data['category'].astype('category')
        self.data['main_category'] = self.data['main_category'].astype(
            'category')
        self.data['currency'] = self.data['currency'].astype('category')
        self.data['state'] = self.data['state'].astype('category')
        self.data['country'] = self.data['country'].astype('category')
        self.data[cat_columns] = self.data[cat_columns].apply(
            lambda x: x.cat.codes)
        Var_Corr = self.data.corr()
        # plot the heatmap and annotation on it
        # sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns,
        #            yticklabels=Var_Corr.columns, annot=True)
        # drop phase 2 according to correlation matrix result
        # remove backers due to high correlation with pladge
        self.data = self.data.drop(['currency', 'category', 'backers'], 1)
        # self.data['usd_pledged_real'].value_counts(normalize=False)
        self.data['goal_level'] = 0  # 0 - low ,1-mid,2 - high,3-very high
        self.data['pledged_level'] = 0  # 0 - low ,1-mid,2 - high,3-very high
        self.data['duration_level'] = 0  # 0 - short-mid , 1-mid-long ,
        # goal bins according distribution
        self.data.loc[(self.data['usd_goal_real'] <= 2000), 'goal_level'] = 0
        self.data.loc[((self.data['usd_goal_real'] > 2000) & (
            self.data['usd_goal_real'] <= 5000)), 'goal_level'] = 1
        self.data.loc[((self.data['usd_goal_real'] > 5000) & (
            self.data['usd_goal_real'] <= 15000)), 'goal_level'] = 2
        self.data.loc[(self.data['usd_goal_real'] > 15000), 'goal_level'] = 3
        # pledges bins
        self.data.loc[(self.data['usd_pledged_real'] <= 5),
                      'pledged_level'] = 0
        self.data.loc[((self.data['usd_pledged_real'] > 5) & (
            self.data['usd_pledged_real'] <= 788)), 'pledged_level'] = 1
        self.data.loc[((self.data['usd_pledged_real'] > 788) & (
            self.data['usd_pledged_real'] <= 4608)), 'pledged_level'] = 2
        self.data.loc[(self.data['usd_pledged_real'] > 4608),
                      'pledged_level'] = 3
        # duration bins
        self.data.loc[(self.data['duration'] <= 30), 'duration_level'] = 0
        self.data.loc[(self.data['duration'] > 30), 'duration_level'] = 1
        # phase 3 of drops results of bins creation
        self.data = self.data.drop(
            ['usd_pledged_real', 'usd_goal_real', 'duration'], 1)
        # reorder columns
        self.data = self.data[['main_category', 'country', 'goal_level',
                               'pledged_level', 'duration_level', 'state']]
        data_Arr = self.data.values
        columns_names = list(self.data.columns)
        # list with tuple (column,unique value range)
        columns_names_with_unique_range = []
        for x in columns_names:
            columns_names_with_unique_range.append(
                (x, len(self.data[x].unique())))

        return data_Arr, columns_names_with_unique_range

    # got numpy array and split to train and test with respect to ratio parm
