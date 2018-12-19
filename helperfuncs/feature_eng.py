import numpy as np
###############################################################################
###############################################################################
# CATEGORICAL FEATURE ENCODING
# The aim is to turn categorical features into numeric features to provide more
# fine-grained information
###############################################################################
class fit_categorical:
    def __init__(self,train_df,cat_var,target):
        self.train_df = train_df
        self.cat_var = cat_var
        self.target = target
        
        self.label_enc = {}
        self.freq_enc = {}
        self.trgt_mean_enc = {}
        self.woe = {}
        self.info_value = {}
    
    def label_encoding(self):
        df = self.train_df.copy()
        for i in self.cat_var:
            cnt = 1
            for j in df[i].unique():
                df[i] = np.where(df[i] == j,cnt,df[i])
                self.label_enc.update({(i,j):cnt})
                cnt +=1
        return df
   
    def freq_encoding(self):
        ''' Good for tree based ensembles.
        We fit the frequency of the levels in each categorical attribute'''
        df = self.train_df.copy()
        for i in self.cat_var:
            for j in df[i].unique():
                freq = len(df[df[i]==j])/len(df)
                df[i] = np.where(df[i] == j,freq,df[i])
                self.freq_enc.update({(i,j):freq})
        return df
    
    def target_mean_encoding(self,k=2,f=0.25,clas = 1,problem = 'reg'):
        ''' Have the mean of the target score for each label. We however smooth
        it so that the mean of the of less frequent categories to reflect that
        of the overall dataset i.e.
        
        lamda(n) * mean(level) + (1-lamda(n))*mean(dataset)
        where lamda(n) = 1/ (1+exp(-(x-k)/f)
        x = frequency of level
        k = inflection point
        f = steepness
        
        We add some uniform noise to limit overfitting/leakage'''
        df = self.train_df.copy()
        if problem == 'reg':
            mean_d = df[self.target].mean()
        else:
            mean_d = len(df[df[self.target]==clas])/len(df)
        for i in self.cat_var:
            for j in df[i].unique():
                x = len(df[df[i]==j])
                lamda = 1/(1+np.exp(-(x-k)/f))
                if problem == 'reg':
                    mean_att = df[self.target][df[i]==j].mean()
                else:
                    mean_att = len(df[df[self.target == clas]][df[i]==j])/x
                score = lamda*mean_att + (1-lamda)*mean_d
                # Add uniform random noise to minimize overfitting
                u_random = np.random.uniform(0,0.1,len(df))
                df[i] = np.where(df[i] == j,score+ u_random,df[i]) 
                self.trgt_mean_enc.update({(i,j):score})
        return df
    
    def weight_of_evidence(self):
        ''' This is good for classification tasks
        '''
        df = self.train_df.copy()
        for i in self.cat_var:
            clas1 = len(df[df[self.target==1]])
            clas2 = len(df[df[self.target==0]])
            for j in df[i].unique():
                clas1_att = len(df[df[self.target==1]][df[i]==j])
                clas2_att = len(df[df[self.target==0]][df[i]==j])
                score = ((clas1_att+0.5)/clas1)/((clas2_att+0.5)/clas2)
                score = np.log(score)
                df[i] = np.where(df[i] == j,score,df[i])
                self.woe.update({(i,j):score})
        return df
    
    def information_value(self):
        ''' This is moslty for feature selection. It takes into account the
        difference between non events and events weighted by w.o.e
        RULE OF THUMB:
            <0.02 Not useful for prediction
            0.02 - 0.1 Weak predictive power
            0.1 - 0.3 Medium predictive power
            0.3 - 0.5 Strong predictive power
            >0.5 Suspect predictive power
        '''
        df = self.train_df.copy()
        for i in self.cat_var:
            clas1 = len(df[df[self.target==1]])
            clas2 = len(df[df[self.target==0]])
            for j in df[i].unique():
                clas1_att = len(df[df[self.target==1]][df[i]==j])
                clas2_att = len(df[df[self.target==0]][df[i]==j])
                score = ((clas1_att+0.5)/clas1)/((clas2_att+0.5)/clas2)
                score = np.log(score) * ((clas1_att/clas1)-(clas2_att/clas2))
                df[i] = np.where(df[i] == j,score,df[i])
                self.info_value.update({(i,j):score})
        return df
###############################################################################
# Using fitted encoders
###############################################################################
    def f_label_encoding(self,test_df):
        df = test_df
        for i in self.cat_var:
            for j in df[i].unique():
                df[i] = np.where(df[i] == j,self.label_enc[i,j],df[i])
        return df
 
    def ff_req_encoding(self,test_df):
        df = test_df
        for i in self.cat_var:
            for j in df[i].unique():
                df[i] = np.where(df[i] == j,self.freq_enc[i,j],df[i])
        return df
    
    def f_target_mean_encoding(self,test_df):
        df = test_df
        for i in self.cat_var:
            for j in df[i].unique():
                df[i] = np.where(df[i] == j,self.trgt_mean_enc[i,j],df[i])
        return df
    
    def f_weight_of_evidence(self,test_df):
        df = test_df
        for i in self.cat_var:
            for j in df[i].unique():
                df[i] = np.where(df[i] == j,self.woe[i,j],df[i])
        return df
    
    def f_information_value(self,test_df):
        df = test_df
        for i in self.cat_var:
            for j in df[i].unique():
                df[i] = np.where(df[i] == j,self.info_value[i,j],df[i])
        return df
    
###############################################################################
# NUMERIC FEATURE ENCODING
###############################################################################