import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args['fileName'])

    def split_train_test(self):
        msk = np.random.rand(len(self.data)) < self.args['trainSize']
        train = self.data[msk]
        test = self.data[~msk]
        return train, test
    # get dataframe and return numpy arrey and columns names and range list

    def preprocess(self):
        """  self.data['deadline'] = self.data['deadline'] = pd.to_datetime(
            self.data['deadline'], format="%Y/%m/%d").dt.date
        self.data['launched'] = self.data['launched'] = pd.to_datetime(
            self.data['launched'], format="%Y/%m/%d").dt.date
        self.data['duration'] = (
            self.data['deadline'] - self.data['launched']).dt.days
        """

        # ignore live\cancel\other because lower density
        self.data = self.data.loc[self.data['state'].isin(
            ['failed', 'successful'])]
        # drop na values
        self.data = self.data.dropna()
        # self.data['country'].value_counts(normalize=True)
        # => US and gb is almost 80% of the self.data =>dimension reduction
        self.data.loc[~self.data['country'].isin(
            ['US', 'GB']), 'country'] = 'OTHER'
        self.data['cat_sub_cat'] = self.data['main_category'] + \
            '_'+self.data['category']
        # feature removal
        self.data = self.data.drop(['ID', 'goal', 'pledged', 'usd pledged',
                                    'name', 'deadline', 'launched', 'category', 'main_category', 'backers', 'usd_pledged_real'], 1)

        cat_columns = ['cat_sub_cat', 'state', 'country', 'month_launched']
        # convert to categorial var
        # ratio  of  mean pledged of the same cat_Sub_cat before the project started
        self.data['ratio'] = (self.data['pledged_s']) / \
            (self.data['usd_goal_real'])
        # log transformation of usd_goal_real
        self.data['goal_log'] = np.log2(self.data['usd_goal_real'])
        self.data['int_log'] = self.data.goal_log.astype(int)

        self.data['cat_sub_cat'] = self.data['cat_sub_cat'].astype('category')
        self.data['currency'] = self.data['currency'].astype('category')
        self.data['state'] = self.data['state'].astype('category')
        self.data['country'] = self.data['country'].astype('category')
        self.data['month_launched'] = self.data['month_launched'].astype(
            'category')
        self.data[cat_columns] = self.data[cat_columns].apply(
            lambda x: x.cat.codes)
        #Var_Corr = self.data.corr()
        # plot the heatmap and annotation on it
        # sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns,
        #            yticklabels=Var_Corr.columns, annot=True)
        # remove backers due to high correlation with pladge
        #self.data = self.data.drop(['currency'], 1)
        # self.data['usd_pledged_real'].value_counts(normalize=False)
        self.data['goal_level'] = 0  # 0 - low ,1-mid,2 - high,3-very high
        self.data['pledged_level'] = 0  # 0 - low ,1-mid,2 - high,3-very high
        self.data['duration_level'] = 0  # 0 - short-mid , 1-mid-long ,
        self.data['ratio_level'] = 0
        # goal bins according distribution
        self.data.loc[(self.data['usd_goal_real'] <=
                       self.data.usd_goal_real.quantile(0.1)), 'goal_level'] = 0
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.1)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.2))), 'goal_level'] = 1
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.2)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.3))), 'goal_level'] = 2
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.3)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.4))), 'goal_level'] = 3
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.4)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.5))), 'goal_level'] = 4
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.5)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.6))), 'goal_level'] = 5
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.6)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.7))), 'goal_level'] = 6
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.7)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.8))), 'goal_level'] = 7
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.8)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.9))), 'goal_level'] = 8
        self.data.loc[(self.data['usd_goal_real'] >
                       self.data.usd_goal_real.quantile(0.9)), 'goal_level'] = 9
        # pledges bins
        """self.data.loc[(self.data['usd_pledged_real'] <= 5),
                      'pledged_level'] = 0
        self.data.loc[((self.data['usd_pledged_real'] > 5) & (
            self.data['usd_pledged_real'] <= 788)), 'pledged_level'] = 1
        self.data.loc[((self.data['usd_pledged_real'] > 788) & (
            self.data['usd_pledged_real'] <= 4608)), 'pledged_level'] = 2
        self.data.loc[(self.data['usd_pledged_real'] > 4608),
                      'pledged_level'] = 3
        """
        # duration bins
        self.data.loc[(self.data['duration'] <= 30), 'duration_level'] = 0
        self.data.loc[(self.data['duration'] > 30), 'duration_level'] = 1

        # ratio bins
        self.data.loc[(self.data['ratio'] <= 0.5), 'ratio_level'] = 0
        self.data.loc[((self.data['ratio'] > 0.5) & (
            self.data['ratio'] <= 1)), 'ratio_level'] = 1
        self.data.loc[(self.data['ratio'] > 1), 'ratio_level'] = 2
        # phase 3 of drops results of bins creation
        """self.data = self.data.drop(
            ['usd_pledged_real', 'usd_goal_real', 'duration', 'ratio'], 1)
        """

        # outlyers removal
        self.data = self.data[self.data['usd_goal_real'] > 20.0]
        self.data = self.data[self.data['ratio'] < 100]
        # date final choose
        self.data = self.data[['cat_sub_cat', 'country', 'goal_level',
                               'duration_level', 'month_launched', 'ratio_level', 'state']]

        # balance data:
        """g=self.data.groupby('state')
        self.data=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))   
        self.data[cat_columns] = self.data[cat_columns].apply(lambda x: x.cat.codes)
        """

        data_Arr = self.data.values
        columns_names = list(self.data.columns)
        # list with tuple (column,unique value range)
        columns_names_with_unique_range = []
        for x in columns_names:
            columns_names_with_unique_range.append(
                (x, len(self.data[x].unique())))

        return data_Arr, columns_names_with_unique_range

    # got numpy array and split to train and test with respect to ratio parm
