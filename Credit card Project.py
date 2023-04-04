#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
pd.options.display.max_columns=50


# In[2]:


df= pd.read_csv('Credit card transactions - India - Simple.csv')


# In[3]:


df.head()


# In[4]:


df.describe(percentiles=[0.75,0.8,0.90,0.95,0.99])


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# ### There are no missing values in our dataset.

# ## Data Cleaning

# In[7]:


df.Date=pd.to_datetime(df['Date'])


# In[8]:


df.info()


# In[9]:


import datetime as dt


# In[10]:


month = df['Date'].dt.month_name()


# In[11]:


df["Day"] = df['Date'].dt.day_name()


# In[12]:


df["Weekday"]= df['Date'].dt.weekday


# In[13]:


year= df.Date.dt.year


# In[14]:


df=pd.concat([df,month,year],axis=1)


# In[15]:


df.head()


# In[16]:


df.columns= ["index","city","Date","Card Type","Exp Type","Gender","Amount","Day","Weekday","Month","Year"]


# In[17]:


df.head()


# In[18]:


df.drop(["index","Date"], axis=1, inplace= True)


# In[19]:


df.head()


# In[20]:


df[["City","Country"]]=df.city.str.split(",",expand=True)


# In[21]:


df.head()


# In[22]:


df.drop(['city','Country'], axis = 1, inplace=True)


# In[23]:


df.head()


# In[24]:


df["Card Type"].value_counts()


# In[25]:


df['Exp Type'].value_counts()


# In[26]:


df.Gender.value_counts()


# In[27]:


df.Year.value_counts()


# In[28]:


df.Month.value_counts()


# ## Identifying and Handling outliers

# In[29]:


df.Amount.describe()


# In[30]:


df[df.Amount>385000]


# In[31]:


sns.boxplot(df.Amount)
plt.show()


# There are no outliers in our Data

# ## Univariate Analysis

# In[32]:


x = ["Silver","Signature","Platinum","Gold"]
y= df["Card Type"].value_counts()
print (y)
plt.figure(figsize=(6,6))
plt.pie(y, labels=x, shadow=True,autopct= "%0.0f%%",textprops={"fontsize":20}, wedgeprops={"width":0.6},explode=(0.1,0,0.1,0))
plt.legend(loc="center right", bbox_to_anchor=(1.3,0.6));


# In[33]:


print(df["Exp Type"].value_counts())
plt.figure(figsize=(9,9))
sns.countplot(x="Exp Type",hue="Exp Type",data= df,edgecolor="black")
plt.legend(loc="center right" ,bbox_to_anchor=(1.3,0.6));


# In[34]:


df.head()


# In[35]:


x=["Female","Male"]
y=df.Gender.value_counts()
print(y)
plt.figure(figsize=(8,8))
plt.pie(y, labels=x,shadow=True,autopct="%0.0f%%",textprops={"fontsize":20,"color":"black"},colors=("pink","blue"),wedgeprops={"width":0.7})
plt.legend(loc="center right",bbox_to_anchor=(1.3,0.8));


# In[36]:


x=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
y=df.Day.value_counts()
plt.figure(figsize=(6,6))
sns.countplot(x="Day",data=df,edgecolor="black")
plt.xticks(rotation=45)


# In[37]:


df.Month.value_counts()
plt.figure(figsize=(6,6))
sns.countplot(x="Month",data=df,edgecolor="black")
plt.xticks(rotation=(90))


# In[38]:


df.City.value_counts().head(10).to_frame()


# In[39]:


df["Amount"].agg(["sum","min","max","mean"]).reset_index()


# # Bivariate Analysis

# In[40]:


df.sample(1)


# ### Checking for corerlation

# In[41]:


sns.heatmap(df.corr(),annot=True,cmap="coolwarm")


# There is no strong corelation 

# In[42]:


df.plot.scatter(x="Gender",y="Amount")


# In[43]:


plt.scatter(df.Gender,df.Amount)


# In[44]:


sns.pairplot(data=df,vars=["Amount","Weekday","Year"])
plt.show()


# In[45]:


df[["Weekday","Year","Amount"]].corr()


# In[46]:


### Which card type were mostly used by the Males and Females
df.groupby("Gender")["Card Type"].value_counts().to_frame()


# In[47]:


plt.figure(figsize=(8,8))
sns.countplot(x="Gender",hue="Card Type",data=df,edgecolor="black")
#plt.figure(figsize=(8,8))
#sns.countplot(x="Card Type",hue="Gender",data=df,palette=["pink","blue"],edgecolor="black");


# In[48]:


print(df.groupby(['Gender'])['Card Type'].value_counts())
plt.figure(figsize=(8,8))
sns.countplot(x="Gender",hue="Card Type",data=df,edgecolor='black')
plt.legend(loc='center right',bbox_to_anchor=(1.2,0.7))


# Females have used Silver card more whereas Males have used Platinum card more.

# In[ ]:





# In[49]:


print(df.groupby('Gender')['Amount'].sum())
plt.figure(figsize=(8,8))
sns.barplot(x='Gender',y='Amount',data=df,estimator=sum,palette=['pink','blue'],edgecolor='black');


# Females spend more amount on credit card than males 

# In[50]:


print(df.groupby(["Gender","Exp Type"])["Amount"].sum())
plt.figure(figsize=(7,7))
sns.barplot(x="Exp Type",y="Amount",hue="Gender",data=df,estimator=sum,palette=['pink','blue'],edgecolor='black')


# In[51]:


print(df.groupby(['Gender','Month'])['Amount'].sum().sort_values( ascending=False))
plt.figure(figsize=(14,14))
sns.barplot(x='Gender',y="Amount",hue='Month',data=df,estimator=sum,edgecolor='black')


# Females spend more in october while Males spend more in january.
# 

# In[52]:


print(df.groupby(['Gender','Day'])['Amount'].sum().sort_values(ascending=False))
plt.figure(figsize=(8,8))
sns.barplot(x='Gender',y='Amount',hue='Day',data=df,estimator=sum,edgecolor='black')
plt.legend(loc='center right', bbox_to_anchor=(1.3,0.9))


# Both females and males spend more on sunday.

# In[53]:


print(df.groupby(["Gender","Exp Type"])["Amount"].sum().sort_values(ascending=False))
plt.figure(figsize=(14,14))
sns.barplot(x="Exp Type",y="Amount",hue="Gender",data=df,estimator=sum,palette=['pink','blue'],edgecolor="black")


# In[54]:


# What was the montly expense type of  males 
male=df[df['Gender']=='M']
print(male.groupby(['Month','Exp Type'])['Amount'])
plt.figure(figsize=(9,9))
sns.lineplot(x='Month',y='Amount',data=male,hue='Exp Type',estimator=sum,ci=None)


# In[55]:


# What was the montly expense type of both males and females
female=df[df['Gender']=='F']
print(female.groupby(["Month","Exp Type"])["Amount"].sum().sort_values(ascending=False))
plt.figure(figsize=(14,14))
sns.lineplot(x='Month',y='Amount',data=female, hue='Exp Type',ci=None,estimator=sum)
plt.show()


# In[56]:


print(df.groupby(["Month","Exp Type"])["Amount"].sum().sort_values(ascending=False))
plt.figure(figsize=(12,12))
sns.lineplot(x='Month',y='Amount',data=df,hue="Exp Type",ci=None,estimator=sum,dashes=True,style="Exp Type",sort=True)
plt.show()


# In[57]:


df.sample()


# In[58]:


weekday_df=df[df["Weekday"]<6]
weekend_df=df[df["Weekday"]>=6]
print("Weekday spends:",weekday_df["Amount"].sum())
print("Weekend spends:",weekend_df["Amount"].sum())


# In[59]:


print(weekday_df["Exp Type"].value_counts())
plt.figure(figsize=(9,9))
sns.countplot(x="Exp Type",data=weekday_df,edgecolor='black')
plt.show()


# In[60]:


print(weekend_df["Exp Type"].value_counts()) 
plt.figure(figsize=(11,11))
sns.countplot(x="Exp Type",data=weekend_df,edgecolor='black')
plt.show()


# In[61]:


print(weekday_df.groupby(['Gender','Exp Type'])['Amount'].sum())
plt.figure(figsize=(9,9))
sns.barplot(x='Exp Type',y='Amount',hue='Gender',data=weekday_df,edgecolor='black',estimator=sum)
plt.show()


# In[62]:


print(weekend_df.groupby(['Gender','Exp Type'])['Amount'].sum())
plt.figure(figsize=(10,10))
sns.barplot(x='Exp Type',y='Amount',hue='Gender',data=weekend_df,edgecolor='black',estimator=sum)
plt.show()


# In[63]:


x=df.groupby('City')['Amount'].sum().sort_values(ascending=False).reset_index().head(10)
print(x)
plt.figure(figsize=(9,9))
sns.barplot(x='City',y='Amount',data=x,estimator=sum,edgecolor='black')
plt.xticks(rotation=90)
plt.show()


# In[64]:


#Relationship between male and female with amount with respect to day
sns.relplot(x='Day',y='Amount',hue='Gender',data=df,kind='line',ci=None)
plt.xticks(rotation=90)
plt.show()


# In[65]:


print(df.groupby(['Gender','Day'])['Amount'].sum())
sns.barplot(x='Day',y='Amount',hue='Gender',data=df)
plt.xticks(rotation=90)
plt.legend(loc='center right',bbox_to_anchor=(1.3,0.7))
plt.show()


# In[66]:


df.sample()


# In[67]:


#which card type was mostly used by males
male=df[df['Gender']=='M']
l=["Platinum","Silver","Signature","Gold"]
x=male.groupby(['Card Type'])['Amount'].sum().sort_values(ascending=False)
print(x)
plt.figure(figsize=(6,6))
plt.pie(x,labels=l,shadow=True,autopct='%0.01f%%',wedgeprops={'width':0.1},textprops={'fontsize':20})
plt.show()


# In[68]:


#Which card type was mostyly used by Females 
female=df[df['Gender']=="F"]
l=['Silver','Signature','Platinum','Gold']
f=female.groupby(['Card Type'])['Amount'].sum().sort_values(ascending=False)
print(f)
plt.figure(figsize=(7,7))
plt.pie(f,labels=l,shadow=True,autopct='%0.01f%%',textprops={'fontsize':20},wedgeprops={'width':0.1})
plt.show()


# In[69]:


#Monthly distributation of amount by Gender
sns.relplot(x='Month',y='Amount',hue="Gender",data=df,kind="line")
plt.xticks(rotation=90)
plt.show()


# In[70]:


#Top 5 cities males and females spend most amount on credit card
male=df[df['Gender']=='M']
x=male.groupby(['City'])['Amount'].sum().sort_values(ascending=False).head(5)
print(x)
l=['Ahmedabad','Bengaluru','Greater Mumbai','Delhi','Kolkata']
plt.pie(x,labels=l,shadow=True,autopct='%0.01f%%',textprops={'fontsize':20},wedgeprops={'width':0.8},explode=(0.1,0.1,0.1,0.1,0.1))
plt.legend(loc='center right',bbox_to_anchor=(1.9,0.8))
plt.show()


# Males are spending more in Ahmedabad using credit card as compared to any other city.

# In[71]:


female=df[df['Gender']=='F']
x=female.groupby(['City'])['Amount'].sum().sort_values(ascending=False).head(5)
print(x)
l=['Greater Mumbai','Bengaluru','Delhi','Ahmedabad','Kanpur']
plt.pie(x,labels=l,shadow=True,autopct='%0.01f%%',textprops={'fontsize':18},wedgeprops={'width':0.8},explode=(0.1,0.1,0.1,0.1,0.1))
plt.legend(loc=('center right'),bbox_to_anchor=(2,0.8))
plt.show()


# Females are spending more in Greater Mumbai using credit card as compared to any other city.

# In[72]:


#Which Exp type creates huge amount
df.groupby(['Exp Type'])['Amount'].sum().sort_values(ascending=False)
sns.barplot(x='Exp Type',y='Amount',data=df,edgecolor='black',estimator=sum)


# In[73]:


#Which is the min amount with details 
df[df["Amount"]==df['Amount'].min()]


# In[74]:


#Which is the max amount with details 
df[df['Amount']==df['Amount'].max()]


# In[75]:


get_ipython().system('pip install wordcloud')


# In[76]:


from wordcloud import WordCloud as word
d=df["City"].value_counts()
wc= word(background_color='white',height=1000,width=600)
wc.generate_from_frequencies(d)


# In[77]:


plt.figure(figsize=(15,10),dpi=100)
plt.imshow(wc)
plt.axis('off')
plt.show()


# Credit spends were more in the highlited cities

# In[78]:


df.to_csv("Credit Card Spending Habits (EDA).csv")


# ##                                                     Insights

# * Mostly Silver card types were used for payments.
# * In Expense types Fuel and Food expenses have recorded most counts paid by the cards.
# *  as per month wise January and December have recorded high usage of cards payments and by day wise sundays have recorded more    no. of card payments.
# * Bengaluru Greater Mumbai, Ahmedabad, Delhi These are the Top 4 Cities where card usage were more.
# * Females have used the Silver card type more where as males have used the platinum card types more.
# * Females are more dependent on credit cards as they spend more amount through credit cards than men.
# * Both Females and Males mostly spend amount on paying Bills and Food using credit card.
# * Females mostly spend in October and Males in january using credit cards.
# * Males have paid highest in january in Fuel expense where as females have paid highest in october in Bills.
# * Weekday Amount spends were more than weekends and that were mostly on Food and entertainment by each gender.
# * Males spend the highest amount with Platinum card type and Females spend the highest amount with Silver card type.
# * Females are spending more in Greater Mumbai where as males are spending more in Ahmedabad.
# * In bills payment the amount spent is the highest.

# In[ ]:





# In[ ]:




