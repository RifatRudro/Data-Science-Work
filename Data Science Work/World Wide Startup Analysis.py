#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
print("Setup Complete")


# In[2]:


df = pd.read_csv('World_Wide_Unicorn_Startups.csv')
df.head()


# In[3]:


df.columns


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.nunique()


# In[7]:


print('Missing Entries:',df.isna().sum().sum())
print('Data Type of Sales Data:\n',df.dtypes, '\n')


# In[8]:


#Checking for the outlier
plt.subplot(3,1,1)
sns.boxplot(Valuation=df.Company, color='#DC143C')

plt.subplot(3,1,3)
sns.boxplot(Valuation=df.Country, color='#15B01A')


# In[9]:


X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)
print((X_train.shape,X_test.shape,y_train.shape,y_test.shape))
type(X_train)


# In[11]:


(m,n) = X_train.shape


# In[12]:


theta=np.zeros((n,1))
theta


# In[13]:


df.head()


# In[14]:


df.Industry.value_counts()
df["Industry"]=df["Industry"].apply(lambda x: "Fintech" if x=="Finttech" else x)
df["Industry"]=df["Industry"].apply(lambda x: "Artificial Intelligence" if x=="Artificial intelligence" else x)


# In[15]:


df.Industry.value_counts().plot(kind="bar",figsize=(19,3))
plt.title("Industry with Most to Least Startups", fontdict={"fontsize":20})


# In[16]:


df.Country.value_counts().head(10)
top_10_country=df.Country.value_counts().head(10)


# In[17]:


top_10_country=pd.DataFrame(top_10_country) 


# In[18]:


top_10_country.columns


# In[19]:


top_10_country=top_10_country.reset_index()
top_10_country.rename(columns={"index":"Country","Country":"No of Startups"},inplace=True)


# In[20]:


plt.figure(figsize=(10,5))
sns.barplot(x="Country",y="No of Startups", data=top_10_country,)
plt.title("Countries and their Number of startups", fontdict={"fontsize":20})
plt.xticks(rotation=75)
plt.show()


# In[21]:


df.Valuation.value_counts().head(10)


# In[22]:


Value50P=df[df.Valuation>50]
Value20P=df[df.Valuation>20]
Value10To20=df[(df.Valuation<20) & (df.Valuation>=10)]
Value5To10=df[(df.Valuation<10) & (df.Valuation>=5)]
Value2To5=df[(df.Valuation<5) & (df.Valuation>=2)]
Value1To2=df[(df.Valuation<2) & (df.Valuation>=1)]

Valuation_5More=df[df.Valuation>5]


# In[23]:


plt.figure(figsize=(17,10))

plt.subplot(2,3,1)
sns.scatterplot(x="Valuation",y="Company",data=Value50P, color="Grey", s=1000, alpha=0.5,edgecolor="black")
plt.title("Valuation>50 companies", fontdict={"fontsize":15})
plt.subplot(2,3,2)
sns.scatterplot(x="Valuation",y="Company",data=Value20P , color="Blue", s=600, alpha=0.5,edgecolor="grey")
plt.title("Valuation>20 companies", fontdict={"fontsize":15})

plt.subplot(2,3,3)
sns.scatterplot(x="Valuation",y="Company",data=Value10To20 , color="cyan", s=600, edgecolor="blue")
plt.title("Valuation 10 to 20 companies", fontdict={"fontsize":15})
plt.subplot(2,3,4)
sns.scatterplot(x="Valuation",y="Company",data=Value5To10.head(30) , color="green", s=600, alpha=0.5, edgecolor="green")
plt.title("Valuation 5 to 10 \nShowing only 30 of 82 companies", fontdict={"fontsize":15})

plt.subplot(2,3,5)
sns.scatterplot(x="Valuation",y="Company",data=Value2To5.head(30) , color="khaki", s=600, alpha=0.5, edgecolor="black")
plt.title("Valuation 2 to 5 \nShowing only 30 of 288 companies", fontdict={"fontsize":15})

plt.subplot(2,3,6)
sns.scatterplot(x="Valuation",y="Company",data=Value1To2.head(30) , color="pink", s=600, alpha=0.5, edgecolor="red")
plt.title("Valuation 1 to 2 \nShowing only 30 of 508 companies", fontdict={"fontsize":15})

plt.tight_layout()


print("Number of companies with Valuation above 50 are         {0}".format(Value50P.Company.nunique()))
print("Number of companies with Valuation between 20 to 50 are {0}".format(Value20P.Company.nunique()))
print("Number of companies with Valuation between 10 to 20 are {0}".format(Value10To20.Company.nunique()))
print("Number of companies with Valuation between 5 to 10 are  {0}".format(Value5To10.Company.nunique()))
print("Number of companies with Valuation between 2 to 5 are   {0}".format(Value2To5.Company.nunique()))
print("Number of companies with Valuation between 1 to 2 are   {0}".format(Value1To2.Company.nunique()))
print("\n \n")


# In[24]:


px.scatter(x="Valuation",y="Company",data_frame=df, color="Country",title="Companies and their valuation countrywise")


# In[25]:


px.scatter_matrix(dimensions=['Valuation', 'year', 'Country',"Industry"],size="Valuation",data_frame=Value20P,color="Company", title="Details of startups between 20 to 50 Valuation vs Country/Year/Industry")


# In[26]:


aTemp=top_10_country["Country"].head(3)
aTemp=list(aTemp)
aTemp


# In[27]:


Country_top3=df[df["Country"].isin(aTemp)]


# In[28]:


Country_top3.columns


# In[29]:


px.scatter(y="Valuation",x="year",data_frame=Country_top3, color="Country",size="Valuation",hover_data=['Company', 'Valuation', 'Date', 'Country', 'City', 'Industry'])


# In[30]:


px.bar(y="Valuation",x="Industry",data_frame=Country_top3, color="Country",hover_data=['Company', 'Valuation', 'Date', 'Country', 'City', 'Industry'])


# In[31]:


px.bar(y="Valuation",x="Industry",data_frame=Country_top3[Country_top3.Country=="India"], color="City",hover_data=['Company', 'Valuation', 'Date', 'Country', 'City', 'Industry'])


# In[ ]:




