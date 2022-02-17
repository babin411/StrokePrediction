from operator import mod
import pickle
import os
from statistics import mode
from sklearn import impute

from sklearn.metrics import roc_curve
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go

from stroke import imputing_mv

st.set_page_config(page_title='Stroke Prediction', 
                   page_icon='https://topnews.in/healthcare/sites/default/files/styles/large/public/Stroke7.jpg?itok=xInaWFYK', 
                   initial_sidebar_state = 'expanded')


def encode_cat_features(df):
    # one_hot_df = pd.get_dummies(df[['gender', 'ever_married','work_type', 'Residence_type','smoking_status']])
    # new_df = pd.concat([df, one_hot_df], axis=1)
    # new_df.drop(['gender','ever_married','Residence_type','work_type','smoking_status'],axis=1,inplace=True)
    # return new_df
    one_hot_df = pd.get_dummies(df)
    new_df = pd.DataFrame(columns=['age','hypertension', 'heart_disease',
                     'avg_glucose_level','bmi','gender_Male',
                     'ever_married_Yes','Residence_type_Urban',
                     'work_type_Never_worked','work_type_Private',
                     'work_type_Self-employed','work_type_children',
                     'smoking_status_never smoked','smoking_status_smokes'])
    for col in one_hot_df.columns:
        if col in new_df.columns:
            new_df.loc[0, col] = one_hot_df.loc[0,col]
            new_df.replace(to_replace=np.nan,value=0,inplace=True)
    return new_df
    

def standardized_df(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(data=scaled_data, columns=df.columns)
    return scaled_df


def user_entry():
    st.sidebar.title("Provide patient's details.")
    gender = st.sidebar.selectbox( "Gender",("Male", "Female"))
    age = st.sidebar.slider('Age',0.0,100.0,step=0.1)
    hypertension = st.sidebar.selectbox('Hypertension',options = (1,0))
    heart_disease = st.sidebar.selectbox('Heart Disease',options = (1,0))
    ever_married = st.sidebar.selectbox('Ever Married?',options= ('Yes','No'))
    work_type = st.sidebar.selectbox('Work Type?',options = (
        'Private','Self-employed','Children','Government Job','Never worked'
    ))
    residence_type = st.sidebar.selectbox('Residence Type?',options = (
        'Rural','Urban'
    ))
    glucose_level = st.sidebar.slider('Blood Glucose Level',0.0,350.0,step=0.1 )
    bmi = st.sidebar.slider('BMI',0.0,100.0, step=0.1)
    smoking_status= st.sidebar.selectbox('Smoking Status',options = (
        'Never Smoked','Unknown', 'Formerly Smoked', 'Smokes'
    ))
    
    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    df = pd.DataFrame(data, index=[0])
    return df


def make_prediction(model,model_name, df):
    prediction = model.predict(df)
    predictions_prob = model.predict_proba(df)
    prediction_label = 'Yes' if prediction==1 else 'No'
    if prediction_label == 'Yes':
        st.subheader(f'Model: {model_name.split(".")[0]}')
        st.subheader(f'Prediction: This person will suffer from stroke.')
    else:
        st.subheader(f'Model: {model_name.split(".")[0]}')
        st.subheader(f'Prediction: This person will not suffer from stroke.')

if __name__=='__main__':
    #setting path to the model
    model_path = os.path.join(os.getcwd(),'models')
    
    header_html = '''
        <h1 style='text-align:center; font-size:56px'>Stroke Prediction</h1>
    '''
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top: -10px" /> """, unsafe_allow_html=True)

    
    option = st.sidebar.selectbox('Choose an Option',options = ('Perform EDA','Make Predictions'))
    
    if option == 'Make Predictions':
        models = st.sidebar.selectbox('Models', 
                options = os.listdir(model_path), index=9)
        df = user_entry()
        if (models != 'svc.pkl') or (models != 'logreg.pkl'):
            final_df = encode_cat_features(df)
        else:
            encoded_df = encode_cat_features(df)
            final_df = standardized_df(encoded_df)
        open_model = open(os.path.join(os.getcwd(), f'models\{models}'), 'rb')
        load_model = pickle.load(open_model)
        
        model_map = {
            'adaboost.pkl': 'AdaBoostClassifier',
            'bnb.pkl': 'BernoulliNB',
            'dtree.pkl': 'DecisionTreeClassifier',
            'gradient_boost.pkl': 'GradientBoostingClassifier',
            'knn.pkl': 'KNeighborsClassifier',
            'logreg.pkl': 'LogisticRegression',
            'rf.pkl': 'RandomForestClassifier',
            'svc.pkl': 'SVC',
            'vc.pkl': 'VotingClassifier',
            'xgboost.pkl': 'XGBClassifier'
        }
        model_eval_df = pd.read_csv(os.path.join(os.getcwd(),'model_evaluation.csv'), index_col='Algorithm')
        model_name = model_map[models]
        print(f'Model Name: {model_name}')
        
        if  st.sidebar.button(label='Predict'):
            make_prediction(load_model,model_name, final_df)
            st.dataframe(data=model_eval_df.loc[model_name, :'ROC-AUC Score'])
            
    if option == 'Perform EDA':
        st.subheader('Original DataFrame From Kaggle')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top: -10px" /> """, unsafe_allow_html=True)
        original_df = pd.read_csv('stroke_data.csv')
        original_df.drop(['id'],axis=1,inplace=True)
        original_df['smoking_status'].replace(to_replace='Unknown',value=np.nan,inplace=True)
        st.dataframe(data=original_df)
        

        st.subheader('Visualizing Missing Data')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top: -10px" /> """, unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax = msno.bar(original_df,sort='descending', color=['black']*9+['red']*2)
        st.pyplot(fig)


        imputed_df = imputing_mv(original_df)  

        st.subheader('Imputed DataFrame')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top: -10px" /> """, unsafe_allow_html=True)
        st.dataframe(data=imputed_df)
        

        st.subheader('Visualizing Missing Data')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top: -10px" /> """, unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax = msno.bar(original_df,sort='descending', color=['black']*11)
        st.pyplot(fig)
        
        st.subheader('Visualizing Data Distribution')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top: -10px" /> """, unsafe_allow_html=True)

        
        #buttons for distribution
        btn1, btn2, btn3 = st.columns(3)
        btn4, btn5, btn6 = st.columns(3)
        btn7, btn8, btn9 = st.columns(3)
        btn10, btn11, btn12 = st.columns(3)
        
        if btn1.button(label='Target Distribution'):
            fig, ax = plt.subplots(1,2, figsize=(20,8))
            sns.countplot(x="stroke", data= imputed_df, palette="tab10",ax=ax[0])
            ax[0].set_title('Countplot of stroke',fontsize=20)
            ax[0].set_xticklabels(['No Stroke','Stroke'], fontsize=14)
            
            
            #Pie Chart of type of gender
            type_counts = imputed_df.stroke.value_counts()
            ax[1].pie(type_counts, labels=['No Stroke','Stroke'],startangle=150, autopct="%1.2f%%", shadow=True,
                    explode=(0,0.2),
                    colors = sns.color_palette('tab10'),textprops={'fontsize':14})
            plt.title("Pie Chart for 'stroke' distribution", fontsize=20)
            st.pyplot(fig)
            
        if btn2.button(label='Gender Distribution'):
            fig, ax = plt.subplots(1,2, figsize=(20,8))
            sns.countplot(x="gender", data= imputed_df, palette="tab10", hue = "gender", ax=ax[0],hue_order=['Female','Male','Other'])
            ax[0].set(xlabel="Gender", ylabel="Count of specific gender")
            ax[0].set_title('Countplot of gender',fontsize=20)
            #Pie Chart of type of gender
            type_counts = imputed_df.gender.value_counts()
            plt.pie(type_counts, labels = type_counts.index, startangle=0, autopct="%1.2f%%", shadow=True,
                    explode=None,
                    colors = sns.color_palette('tab10'),textprops={'fontsize':20})
            plt.title("Pie Chart for gender distribution", fontsize=20)
            st.pyplot(fig)
            
        if btn3.button(label='Hypertension Distribution'):
            fig, ax = plt.subplots(1,2, figsize=(20,8))
            sns.countplot(x="hypertension", data= imputed_df, palette="tab10", ax=ax[0],)
            ax[0].set(xlabel="Hypertension", ylabel="Counts")
            ax[0].set_title('Countplot of hypertension',fontsize=20)
            ax[0].set_xticklabels(['No Hypertension','Hypertension'], fontsize=14)
            #Pie Chart of type of gender
            type_counts = imputed_df.hypertension.value_counts()
            labels=[f'No\nHypertension', 'Hypertension']
            plt.pie(type_counts, labels = labels, startangle=0, autopct="%1.2f%%", shadow=True,
                    explode=(0,0.2),
                    colors = sns.color_palette('tab10'),textprops={'fontsize':20})
            plt.title("Pie Chart for hypertension distribution", fontsize=20)
            st.pyplot(fig)
            
        if btn4.button(label='Heart Disease Distribution'):
            fig, ax = plt.subplots(1,2, figsize=(20,8))
            sns.countplot(x="heart_disease", data= imputed_df, palette="tab10", ax=ax[0],)
            ax[0].set(xlabel="heart_disease", ylabel="Count of heart_disease")
            ax[0].set_title('Countplot of heart_disease',fontsize=20)
            ax[0].set_xticklabels(['No heart_disease','heart_disease'], fontsize=14)
            #Pie Chart of type of gender
            type_counts = imputed_df.heart_disease.value_counts()
            labels = [f'No\nHeart Disease', 'Heart\nDisease']
            plt.pie(type_counts, labels = labels, startangle=0, autopct="%1.2f%%", shadow=True,
                    explode=(0,0.2),
                    colors = sns.color_palette('tab10'),textprops={'fontsize':20})
            plt.title("Pie Chart for heart_disease distribution", fontsize=20)
            st.pyplot(fig)
            
            
        if btn5.button(label='Marital Status Distribution'):
            fig, ax = plt.subplots(1,2, figsize=(20,8))
            sns.countplot(x="ever_married", data= imputed_df, palette="tab10", ax=ax[0],)
            ax[0].set(xlabel="ever_married", ylabel="Count of ever_married")
            ax[0].set_title('Countplot of ever_married',fontsize=20)
            ax[0].set_xticklabels(['Yes','No'], fontsize=14)
            #Pie Chart of type of gender
            type_counts = imputed_df.ever_married.value_counts()
            plt.pie(type_counts, labels =['Yes','No'], startangle=0, autopct="%1.2f%%", shadow=True,
                    explode=None,
                    colors = sns.color_palette('tab10'),textprops={'fontsize':20})
            plt.title("Pie Chart for ever_married distribution", fontsize=20)
            st.pyplot(fig)
            
        
        if btn6.button(label='Work Type Distribution'):
            fig, ax = plt.subplots(1,2, figsize=(20,8))
            type_counts = imputed_df.work_type.value_counts()
            sns.countplot(x="work_type", data= imputed_df, palette="tab10", ax=ax[0],order = type_counts.index)
            ax[0].set(xlabel="work_type", ylabel="Count of work_type")
            ax[0].set_title('Countplot of work_type',fontsize=20)

            #Pie Chart of type of ever_married
            plt.pie(type_counts, labels =type_counts.index, startangle=0, autopct="%1.2f%%", shadow=True,
                    explode=(0.1,0,0,0,0),
                    colors = sns.color_palette('tab10'),textprops={'fontsize':20})
            plt.title("Pie Chart for work_type distribution", fontsize=20)
            st.pyplot(fig)
            
            
        if btn7.button(label='Residence Type Distribution'):
            type_counts = imputed_df.Residence_type.value_counts()
            fig, ax = plt.subplots(1,2, figsize=(20,8))
            sns.countplot(x="Residence_type", data= imputed_df, palette="tab10", ax=ax[0],order=type_counts.index)
            ax[0].set(xlabel="Residence_type", ylabel="Count of Residence_type")
            ax[0].set_title('Countplot of Residence_type',fontsize=20)

            #Pie Chart of type of ever_married
            plt.pie(type_counts, labels =type_counts.index, startangle=0, autopct="%1.2f%%", shadow=True,
                    explode=(0.1,0),
                    colors = sns.color_palette('tab10'),textprops={'fontsize':20})
            plt.title("Pie Chart for Residence_type distribution", fontsize=20)
            st.pyplot(fig)
            
        if btn8.button(label='Smoking Status Distribution'):
            type_counts = imputed_df.smoking_status.value_counts()
            fig, ax = plt.subplots(1,2, figsize=(20,8))
            sns.countplot(x="smoking_status", data= imputed_df, palette="tab10", ax=ax[0],order=type_counts.index)
            ax[0].set(xlabel="smoking_status", ylabel="Count of Smoking Status")
            ax[0].set_title('Countplot of Smoking Status',fontsize=20,pad=30)

            #Pie Chart of type of ever_married
            plt.pie(type_counts, labels =type_counts.index, startangle=0, autopct="%1.2f%%", shadow=True,
                    explode=(0.2,0,0),
                    colors = sns.color_palette('tab10'),textprops={'fontsize':20})
            plt.title("Pie Chart for Smoking Status distribution", fontsize=20,pad=30)
            st.pyplot(fig)
            
        
        if btn9.button(label='Stroke vs Age'):
            fig, ax = plt.subplots(1,2,figsize=(10,6))
            sns.kdeplot(x='age',data=imputed_df,shade=True,ax=ax[0])
            ax[0]= plt.title('Distribution of Age')
            sns.kdeplot(x='age',data=imputed_df[imputed_df.stroke==1], shade=True, label='Stroke',ax=ax[1])
            sns.kdeplot(x='age',data=imputed_df[imputed_df.stroke==0], shade=True, label='No Stroke',ax=ax[1])
            ax[1]=plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax[1]=plt.title('Distribution of Age according to Stroke')
            st.pyplot(fig)
            
        
        if btn10.button(label='Stroke vs Glucose Level'):
            fig, ax = plt.subplots(1,2,figsize=(10,6))
            sns.kdeplot(x='avg_glucose_level',data=imputed_df,shade=True,ax=ax[0])
            ax[0]= plt.title('Distribution of Glucose Level')
            sns.kdeplot(x='avg_glucose_level',data=imputed_df[imputed_df.stroke==1], shade=True, label='Stroke',ax=ax[1])
            sns.kdeplot(x='avg_glucose_level',data=imputed_df[imputed_df.stroke==0], shade=True, label='No Stroke',ax=ax[1])
            ax[1]=plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax[1]=plt.title('Distribution of Glucose Level according to Stroke')
            st.pyplot(fig)
            
        if btn11.button(label='Stroke vs BMI'):
            fig, ax = plt.subplots(1,2,figsize=(10,6))
            sns.kdeplot(x='bmi',data=imputed_df,shade=True,ax=ax[0])
            ax[0]= plt.title('Distribution of BMI')
            sns.kdeplot(x='bmi',data=imputed_df[imputed_df.stroke==1], shade=True, label='Stroke',ax=ax[1])
            sns.kdeplot(x='bmi',data=imputed_df[imputed_df.stroke==0], shade=True, label='No Stroke',ax=ax[1])
            ax[1]=plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax[1]=plt.title('Distribution of bmi according to Stroke')
            st.pyplot(fig)
            
        st.subheader('Visualizing Data Distribution')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top: -10px" /> """, unsafe_allow_html=True)

        
        #buttons for distribution
        btn13, btn14, btn15 = st.columns(3)
        cat_features = imputed_df.select_dtypes(['object']).columns.tolist()
    
        if btn13.button(label='Analysis of Categorical Variables with respect to age'):
            fig2, ax = plt.subplots(2,3,figsize=(20,12))
            fig2.delaxes(ax[1][2])
            i=0
            for row in range(2):
                for col in range(3):
                    if i < 5:
                        sns.boxplot(x=cat_features[i], y='age',data=imputed_df, ax=ax[row][col], hue='stroke',)
                        i+=1
            plt.legend(loc='best')
            st.pyplot(fig2)
        
        
        if btn14.button(label='Analysis of Categorical Variables with respect to bmi'):
            fig2, ax = plt.subplots(2,3,figsize=(20,12))
            fig2.delaxes(ax[1][2])
            i=0
            for row in range(2):
                for col in range(3):
                    if i < 5:
                        sns.boxplot(x=cat_features[i], y='bmi',data=imputed_df, ax=ax[row][col], hue='stroke',)
                        i+=1
            plt.legend(loc='best')
            st.pyplot(fig2)
            
        
        if btn15.button(label='Analysis of Categorical Variables with respect to avg_glucose_level'):
            fig2, ax = plt.subplots(2,3,figsize=(20,12))
            fig2.delaxes(ax[1][2])
            i=0
            for row in range(2):
                for col in range(3):
                    if i < 5:
                        sns.boxplot(x=cat_features[i], y='avg_glucose_level',data=imputed_df, ax=ax[row][col], hue='stroke',)
                        i+=1
            plt.legend(loc='best')
            st.pyplot(fig2)
        
    
        st.subheader('Evaluating Model Performance')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;margin-top: -10px" /> """, unsafe_allow_html=True)
        model_eval_df = pd.read_csv(os.path.join(os.getcwd(),'model_evaluation.csv'), index_col='Algorithm')
        model_eval_df = model_eval_df.loc[:,:'ROC-AUC Score']
        model_eval_df.reset_index(inplace=True)
        
        train_model_eval_df = pd.read_csv(os.path.join(os.getcwd(),'model_evaluation_train.csv'), index_col='Algorithm')
        train_model_eval_df.reset_index(inplace=True)
        st.dataframe(model_eval_df.style.highlight_max(color='green'))
        
        btn16, btn17, btn18 = st.columns(3)
        btn19, btn20, btn21 = st.columns(3)
        
        if btn16.button('Accuracy Score'):
            # fig = px.line(model_eval_df,x='Algorithm',
            #         y='Accuracy Score',template='none', markers=True
            #         )
            # fig.add_line(train_model_eval_df, x='Algorithm', y='Accuracy Score',
            #               template='none', markers=True)
            # fig.update_layout(
            #     title="Accuracy Plot",
            #     xaxis_title="Algorithms",
            #     yaxis_title="Accuracy",
            # )
            # st.write(fig)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=model_eval_df['Algorithm'], 
                    y=model_eval_df['Accuracy Score'],
                    name='Test Set'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_model_eval_df['Algorithm'], 
                    y=train_model_eval_df['Accuracy Score'],
                    name='Train Set',
                )
            )
            fig.update_layout(
                title="Accuracy Plot",
                xaxis_title="Algorithms",
                yaxis_title="Accuracy",
            )
            st.write(fig)
            
            
            
            
        if btn17.button('Precision Score'):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=model_eval_df['Algorithm'], 
                    y=model_eval_df['Precision Score'],
                    name='Test Set'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_model_eval_df['Algorithm'], 
                    y=train_model_eval_df['Precision Score'],
                    name='Train Set'
                )
            )
            fig.update_layout(
                title="Prcision Plot",
                xaxis_title="Algorithms",
                yaxis_title="Precision",
            )
            st.write(fig)
            
        if btn18.button('Recall Score'):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=model_eval_df['Algorithm'], 
                    y=model_eval_df['Recall Score'],
                    name='Test Set'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_model_eval_df['Algorithm'], 
                    y=train_model_eval_df['Recall Score'],
                    name='Train Set'
                )
            )
            fig.update_layout(
                title="Recall Plot",
                xaxis_title="Algorithms",
                yaxis_title="Recall",
            )
            st.write(fig)
        
        
        if btn19.button('F1 Score'):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=model_eval_df['Algorithm'], 
                    y=model_eval_df['F1 Score'],
                    name='Test Set'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_model_eval_df['Algorithm'], 
                    y=train_model_eval_df['F1 Score'],
                    name='Train Set'
                )
            )
            fig.update_layout(
                title="F1 Plot",
                xaxis_title="Algorithms",
                yaxis_title="F1",
            )
            st.write(fig)
            
        if btn20.button('ROC-AUC Score'):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=model_eval_df['Algorithm'], 
                    y=model_eval_df['ROC-AUC Score'],
                    name='Test Set'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_model_eval_df['Algorithm'], 
                    y=train_model_eval_df['ROC-AUC Score'],
                    name='Train Set'
                )
            )
            fig.update_layout(
                title="ROC-AUC Plot",
                xaxis_title="Algorithms",
                yaxis_title="ROC-AUC",
            )
            st.write(fig)