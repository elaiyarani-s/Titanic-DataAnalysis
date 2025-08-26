## import libraries
import pandas as pd
import streamlit as st
import pickle
import numpy as np


## Main Function
def main():
    """ main() contains all UI structure elements; getting and storing user data can be done within it"""
    st.title("Titanic Survival Prediction")                                                                              ## Title/main heading
    st.image(r"titanic_sinking.jpg", caption="Sinking of 'RMS Titanic' : 15 April 1912 in North Atlantic Ocean",use_column_width=True) ## image import
    st.write("""## Would you have survived From Titanic Disaster?""")                                                    ## Sub heading

    st.title("-----          Check Your Survival Chances          -----")

    ## Framing UI Structure
    age = st.slider("Age :", 1, 75, 30)                                                                  
    fare = st.slider("Ticket Fare (£) :", 15, 500, 40)                                                        
    SibSp = st.selectbox("Siblings/Spouses aboard: ", [0, 1, 2, 3, 4, 5, 6, 7, 8]) 
    Parch = st.selectbox("Parents/Children aboard: ", [0, 1, 2, 3, 4, 5, 6, 7, 8]) 
    sex = st.selectbox("Select Gender:", ["Male","Female"])                        
    Sex = 0 if sex == "Male" else 1


    Pclass= st.selectbox("Select Passenger-Class:",[1,2,3])                       

    boarded_location = st.selectbox("Boarded Location:", ["Southampton","Cherbourg","Queenstown"]) 
    Embarked_C,Embarked_Q,Embarked_S=0,0,0                     
    if boarded_location == "Queenstown":
        Embarked_Q=1
    elif boarded_location == "Southampton":
        Embarked_S=1
    else:
        Embarked_C=1

    FamilySize = (SibSp if SibSp is not None else 0) + (Parch if Parch is not None else 0) + 1

    data={"Age":age,"Fare":fare,"FamilySize":FamilySize,"Sex":Sex,"Pclass":Pclass,"Embarked_Q":Embarked_Q,"Embarked_S":Embarked_S}

    df=pd.DataFrame(data,index=[0])     
    return df

model_list={
    'Logistic Regression':'models/titanic_Logistic Regression.pkl',
    'Decision Tree Classifier':'models/titanic_Decision Tree.pkl',
    'Random Forest Classifier':'models/titanic_Random Forest.pkl',
    'XGBoost Classifier':'models/titanic_XGBoost.pkl'

}
select_model = st.selectbox('Choose a model', list(model_list.keys()))
model_file = model_list[select_model] # type: ignore
print(model_file)
with open(model_file,'rb') as f:
    model = pickle.load(f)

data=main()                            
print(data)
## Prediction:
if st.button("Predict"):                                                                
    result = model.predict(data)                                                       
    proba=model.predict_proba(data)                                                   
    #st.success('The output is {}'.format(result))

    if result[0] == 1:
        st.write("***congratulation !!!....*** **You probably would have made it!**")
        st.write("**Survival Probability Chances :** 'NO': {}%  'YES': {}% ".format(round((proba[0,0])*100,2),round((proba[0,1])*100,2)))
    else:
        st.write("***Better Luck Next time !!!!...*** **you're probably Ended up like 'Jack'**")
        st.write("**Survival Probability Chances :** 'NO': {}%  'YES': {}% ".format(round((proba[0,0])*100,2),round((proba[0,1])*100,2)))

## Working Button:
if st.button("Working"):                                                      # creating Working button, which gets some survival tips & info.
    st.write("""# How's prediction Working :- Insider Survival Facts and Tips: 
             - Only about `38%` of passengers survived In this Accident\n
             - Ticket price:
                    1st-class: $150-$435 ; 2nd-class: $60 ; 3rd-class: $15-$40\n
             - About Family Factor:
                If You Boarded with atleast one family member `51%` Survival rate
               """)

## Author Info.
# if st.button("Author"):
#     st.write("## @ Elaiyarani S")