#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 


# In[2]:




# In[4]:


df = pd.read_csv(r"C:\Users\perum\OneDrive\Desktop\model2\Auto MPG Reg.csv")


# In[5]:


df


# In[6]:


df.horsepower=pd.to_numeric(df.horsepower,errors="coerce")


# In[7]:


df.horsepower = df.horsepower.fillna(df.horsepower.median())


# In[8]:


#split data 
y = df.mpg
X = df.drop(['carname','mpg'],axis = 1)


# In[14]:


# Defining Multiple Models as a "Dictionary"
models={'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(),'Random Forest':RandomForestRegressor(),
        'Gradient Boosting':GradientBoostingRegressor()}


# In[15]:


selected_model=st.sidebar.selectbox("Select a ML model", list(models.keys()))


# In[16]:


# ML model Selection Parameters
if selected_model=='Linear Regression':
    model=LinearRegression()
elif selected_model=='Decision Tree':
    max_depth=st.sidebar.slider("max_depth",8,16,2)
    model=DecisionTreeRegressor(max_depth=max_depth)
elif selected_model=='Random Forest':
    n_estimators=st.slider.sidebar("Num of Trees",1,100,10)
    model=RandomForestRegressor(n_estimators=n_estimators)
elif selected_model=='Gradient Boosting':
    n_estimators=st.sidebar.slider("Num of Trees",1,100,10)
    model=GradientBoostingRegressor(n_estimators=n_estimators)


# In[17]:


# Train the model 
model.fit(X,y)


# In[20]:


#Define the application page parameters
st.title("Predict Mileage Per Gallon")
st.markdown("Model to Predict Mileage of Car")
st.header("Car Features")

col1,col2,col3,col4 = st.columns(4)
with col1:
    cylinders = st.slider("Cylinders",2,8,1)
    displacement = st.slider("Displacement",50,500,10)
with col2:
    horsepower = st.slider("HorsePower",50,500,10)
    weight = st.slider("Weight",1500,5000,250)
with col3:
    acceleration = st.slider("Accel",8,25,1)
    modelyear = st.slider("year",70,85,1)
with col4:
    origin = st.slider("Orif=gin",1,3,1)


# In[21]:


# Model Preditions
rsquare = model.score(X,y)
y_pred = model.predict(np.array([[cylinders,displacement,horsepower,weight,acceleration,modelyear,origin]]))


# In[23]:


#Display Results
st.header('ML Model Results')
st.write(f"Selected Model: {selected_model}")
st.write(f"RSquare:{rsquare}")
st.write(f"Predicted:{y_pred}")


# In[ ]:




