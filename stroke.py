from distutils.log import Log
from hashlib import new
import os
import time 
import numpy as np
import pandas as pd

#importing preprocessing modules
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

#importing models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier

#import evluating modules
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix


#importing plotting libraries
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.graph_objects as go

#saving model
import pickle


def import_dataset():
    df  = pd.read_csv('stroke_data.csv')
    df.drop(['id'], axis=1, inplace=True)
    return df


def imputing_mv(df):
    #Replacing 'Unknown' with Nan values
    df['smoking_status'].replace(to_replace='Unknown',value=np.nan,inplace=True)

    female_rural = df.groupby(['gender','Residence_type'])['bmi'].mean().values[0]
    female_urban = df.groupby(['gender','Residence_type'])['bmi'].mean().values[1]
    male_rural = df.groupby(['gender','Residence_type'])['bmi'].mean().values[2]
    male_urban = df.groupby(['gender','Residence_type'])['bmi'].mean().values[3]

    #Setting all the null values in  bmi  
    df.loc[((df.bmi.isna()) & (df.gender=='Female') & (df.Residence_type=='Rural').values), 'bmi'] = female_rural
    df.loc[((df.bmi.isna()) & (df.gender=='Female') & (df.Residence_type=='Urban').values), 'bmi'] = female_urban
    df.loc[((df.bmi.isna()) & (df.gender=='Male') & (df.Residence_type=='Rural').values), 'bmi'] = male_rural
    df.loc[((df.bmi.isna()) & (df.gender=='Male') & (df.Residence_type=='Urban').values), 'bmi'] = male_urban

    #Replacing 'Unknown' with Nan values
    df['smoking_status'].replace(to_replace=np.nan, value='never smoked',inplace=True)
    return df


def encode_cat_features(df):
    df.drop(index=3116, inplace=True)
    df.reset_index(inplace=True)
    one_hot_df = pd.get_dummies(df[['gender','ever_married','Residence_type','work_type','smoking_status']],drop_first=True)
    new_df = pd.concat([df,one_hot_df],axis=1)
    new_df.drop(['index','gender','ever_married','Residence_type','work_type','smoking_status'],axis=1,inplace=True)
    return new_df


def handling_imbalanced_dataset(df):
    smote = SMOTE(sampling_strategy='minority')
    X,y = smote.fit_resample(df.loc[:, df.columns!='stroke'],df['stroke'])
    return X,y
    
    
def models(X_train,y_train):
    classifiers = []
    accuracy, precision, recall, f1, confusion_mat,auc_score = [],[],[],[],[],[]
    
    train_accuracy, train_precision, train_recall, train_f1, train_auc_score = [],[],[],[],[]
    
    classifiers.append(LogisticRegression(solver='lbfgs'))
    classifiers.append(SVC(kernel='rbf',probability=True))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', 
                                              max_depth=2,
                                              max_features=None,
                                              splitter='best'))
    classifiers.append(KNeighborsClassifier(metric='euclidean',
                                            n_neighbors=1,
                                            weights='uniform'))
    classifiers.append(RandomForestClassifier(max_features='log2'))
    classifiers.append(BernoulliNB(alpha=0.01))
    classifiers.append(VotingClassifier(voting='soft',estimators=[
    ('logreg', LogisticRegression(solver='lbfgs')),
    ('knn', KNeighborsClassifier(metric='euclidean', n_neighbors=2, weights='uniform')),
    ('dt',DecisionTreeClassifier(criterion='entropy', max_depth=25, max_features=None, splitter='best'))
]))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier()))
    classifiers.append(GradientBoostingClassifier(loss='deviance'))
    classifiers.append(XGBClassifier(objective='reg:logistic',
                                     learning_rate = 0.25,
                                    colsample_bytree=0.4,
                                    gamma=0.3,
                                    max_depth = 14,
                                    min_child_weight=1))
    
    i = 0
    for classifier in classifiers:
        model_name = ['logreg', 'svc','dtree','knn','rf','bnb','vc','adaboost','gradient_boost','xgboost']
        
        start_time = time.time()
        print(f'Fitting on Classifier: {classifier}')
        #fitting the model
        classifier.fit(X_train,y_train)
        
        #making predictions
        predictions = classifier.predict(X_test)
        predictions_prob = classifier.predict_proba(X_test)[:,1]
        
        #evaluating model on test_set
        accuracy.append(accuracy_score(y_test,predictions))
        precision.append(precision_score(y_test,predictions))
        recall.append(recall_score(y_test, predictions))
        f1.append(f1_score(y_test,predictions))
        auc_score.append(roc_auc_score(y_test, predictions_prob))
        
        #confusion matrix
        confusion_mat.append(confusion_matrix(y_test, predictions))
        
        #evaluating model on train_set
        train_predictions = classifier.predict(X_train)
        train_predictions_prob = classifier.predict_proba(X_train)[:,1]
        train_accuracy.append(accuracy_score(y_train, train_predictions))
        train_precision.append(precision_score(y_train, train_predictions))
        train_recall.append(recall_score(y_train, train_predictions))
        train_f1.append(f1_score(y_train, train_predictions))
        train_auc_score.append(roc_auc_score(y_train, train_predictions))
        
        end_time = time.time()
        print(f'Total Training Time: {(end_time - start_time):.2f} seconds')
        
        #saving model
        pickle.dump(classifier,open(os.path.join(model_path,model_name[i])+'.pkl', 'wb'))
        i+=1
        
        
    results_df = pd.DataFrame({
        'Accuracy Score': accuracy,
        'Precision Score': precision,
        'Recall Score': recall,
        'F1 Score': f1,
        'ROC-AUC Score': auc_score,
        'Confusion Matrix': confusion_mat,
        'Algorithm': ['LogisticRegression','SVC', 'DecisionTreeClassifier','KNeighborsClassifier','RandomForestClassifier',
                     'BernoulliNB','VotingClassifier','AdaBoostClassifier','GradientBoostingClassifier','XGBClassifier']
    })
    results_df.set_index('Algorithm',inplace=True)
    
    train_results_df = pd.DataFrame({
        'Accuracy Score': train_accuracy,
        'Precision Score': train_precision,
        'Recall Score': train_recall,
        'F1 Score': train_f1,
        'ROC-AUC Score': train_auc_score,
        'Algorithm': ['LogisticRegression','SVC', 'DecisionTreeClassifier','KNeighborsClassifier','RandomForestClassifier',
                     'BernoulliNB','VotingClassifier','AdaBoostClassifier','GradientBoostingClassifier','XGBClassifier']
    })
    train_results_df.set_index('Algorithm',inplace=True)
    
    results_df.to_csv('model_evaluation.csv')
    train_results_df.to_csv('model_evaluation_train.csv')
    return results_df, train_results_df
            
    
if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(),'models')
    df = import_dataset()
    cleaned_df = imputing_mv(df)
    encoded_df = encode_cat_features(cleaned_df)
    X,y = handling_imbalanced_dataset(encoded_df)    

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    
    results_df = models(X_train,y_train)