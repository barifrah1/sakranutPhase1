import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args['fileName'])  # , nrows=25000)

        # balance data:
        # g=self.data.groupby('state')
        #self.data=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

    # only for evaluation during our development phase
    def split_train_test(self):
        msk = np.random.rand(len(self.data)) < self.args['trainSize']
        train = self.data[msk]
        test = self.data[~msk]
        return train, test
    # get dataframe and return numpy array and columns names and range list

    def preprocess(self):
        # ignore live\cancel\other due to lower density and it's not really intresting
        self.data = self.data.loc[self.data['state'].isin(
            ['failed', 'successful'])]
        # drop na values
        self.data = self.data.dropna()
        # self.data['country'].value_counts(normalize=True)
        # => US and gb is almost 80% of the self.data =>dimension reduction
        self.data.loc[~self.data['country'].isin(
            ['US', 'GB']), 'country'] = 'OTHER'
        # create category with subcategoy column
        self.data['cat_sub_cat'] = self.data['main_category'] + \
            '_'+self.data['category']
        # feature removal
        self.data = self.data.drop(['ID', 'goal', 'pledged', 'usd pledged',
                                    'name', 'deadline', 'launched', 'category', 'main_category', 'backers', 'usd_pledged_real'], 1)

        cat_columns = ['cat_sub_cat', 'state', 'country', 'duration']
        # convert to categorial var
        # ratio  of  mean pledged of the same cat_Sub_cat before the project started
        # log transformation of usd_goal_real
        self.data['goal_log'] = np.log2(self.data['usd_goal_real'])
        # discretization of all other fields
        self.data['cat_sub_cat'] = self.data['cat_sub_cat'].astype('category')
        self.data['currency'] = self.data['currency'].astype('category')
        self.data['state'] = self.data['state'].astype('category')
        self.data['country'] = self.data['country'].astype('category')
        #self.data['month_launched'] = self.data['month_launched'].astype('category')
        self.data['duration'] = self.data['duration'].astype('category')
        self.data[cat_columns] = self.data[cat_columns].apply(
            lambda x: x.cat.codes)

        # goal_level prior calculating
        # data[(data['usd_goal_real']>=data.usd_goal_real.quantile(0.75))
        # &(data['usd_goal_real']<data.usd_goal_real.quantile(1))]['state'].value_counts(normalize=True)
        # corr matrix calculation to remove features
        #Var_Corr = self.data.corr()
        # plot the heatmap and annotation on it
        # sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns,
        #            yticklabels=Var_Corr.columns, annot=True)

        # remove backers due to high correlation with pledge
        self.data['goal_level'] = 0  # 0 - low ,1-mid,2 - high,3-very high
        # self.data['pledged_level'] = 0  # 0 - low ,1-mid,2 - high,3-very high
        self.data['duration_level'] = 0  # 0 - short-mid , 1-mid-long ,
        # goal bins according distribution quantiles
        self.data.loc[(self.data['usd_goal_real'] <=
                       self.data.usd_goal_real.quantile(0.25)), 'goal_level'] = 0
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.25)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.5))), 'goal_level'] = 1
        self.data.loc[((self.data['usd_goal_real'] > self.data.usd_goal_real.quantile(0.5)) & (
            self.data['usd_goal_real'] <= self.data.usd_goal_real.quantile(0.75))), 'goal_level'] = 2
        self.data.loc[(self.data['usd_goal_real'] >
                       self.data.usd_goal_real.quantile(0.75)), 'goal_level'] = 3

        # suceussfull projects with average 32 and failed projects average is 36
        self.data.loc[(self.data['duration'] <= 30), 'duration_level'] = 0
        self.data.loc[(self.data['duration'] > 30), 'duration_level'] = 1
        # final features selected
        self.data = self.data[['cat_sub_cat', 'country',
                               'duration_level', 'goal_level', 'state']]

        data_Arr = self.data.values
        columns_names = list(self.data.columns)
        # list with tuple (column,unique value range)
        columns_names_with_unique_range = []
        for x in columns_names:
            columns_names_with_unique_range.append(
                (x, len(self.data[x].unique())))

        return data_Arr, columns_names_with_unique_range
