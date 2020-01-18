#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
#from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import sys


#files = os.listdir("../input")
#input_file = "../input/{}".format(files[0])

#files = os.listdir("globalterrorismdb_0718dist.csv")
input_file = "globalterrorismdb_0718dist.csv"

# Any results you write to the current directory are saved as output.


# In[20]:


# Considering a single region
#region_id = 6
country_id=92
df = pd.read_csv(input_file, header = 0,usecols=['iyear', 'imonth', 'iday', 'extended', 'country', 'country_txt', 'region', 'latitude', 'longitude','success', 'suicide','attacktype1','attacktype1_txt', 'targtype1', 'targtype1_txt', 'natlty1','natlty1_txt','weaptype1', 'weaptype1_txt' ,'nkill','multiple', 'individual', 'claimed','nkill','nkillter', 'nwound', 'nwoundte'])
# Filetered dataframe
#df_Area = df[df.region == region_id]
df_Area = df[df.country == country_id]


#df_Area.describe()
#df_Area.info()


# In[21]:


# Dropping the uneccessary columns
df_Country = df_Area.drop([ 'region', 'claimed', 'nkillter', 'nwound','nwoundte'], axis=1)

# Fill NA
df_Country['nkill'].fillna(df_Country['nkill'].mean(), inplace=True)
df_Country['latitude'].fillna(df_Country['latitude'].mean(), inplace=True)
df_Country['longitude'].fillna(df_Country['longitude'].mean(), inplace=True)
df_Country['natlty1'].fillna(df_Country['natlty1'].mean(), inplace=True)

#df_Country.info()


# In[22]:


# Kill Plot comparison
'''df_Country.plot(kind= 'scatter', x='longitude', y='latitude', alpha=1.0,  figsize=(18,10),
                   s=df_Country['nkill']*3, label= 'Casualties', fontsize=1, c='nkill', cmap=plt.get_cmap("jet"), colorbar=True)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.show()
'''


# In[23]:


# Verify Correlation Matrix
corrmat = df_Country.corr()
#f, ax = plt.subplots(figsize=(10, 10))
#sns.heatmap(corrmat, vmax=1, square=True)


# In[24]:


X = df_Country.drop(['iyear', 'success','country', 'country_txt', 'attacktype1_txt','targtype1_txt','natlty1', 'natlty1_txt', 'weaptype1_txt'], axis=1)
y = df_Country['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)


# In[25]:


classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
#print('Accuracy Score: {}'.format(acc))


# In[26]:


#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#print('Mean: {}'.format(accuracies.mean()))
#print('SD: {}'.format(accuracies.std()))


# In[27]:


# Pass parameters to predict whether an attack is successful or not
# Date params

month = sys.argv[1]
day = sys.argv[2]
# Boolean 0-No,; 1-Yes
extended = sys.argv[3]
# Location
latitude = sys.argv[4]
longitude = sys.argv[5]
# Attack Params
multiple = sys.argv[6]
suicide = sys.argv[7]
attackType = sys.argv[8]
targetType = sys.argv[9]
individual = sys.argv[10]
weaponType = sys.argv[11]
# Aftermath --> Casuality Number
nkill = 0

attack_params = np.array([[(month),(day),(extended),(latitude),(longitude),(multiple),(suicide),(attackType),(targetType),(individual),(weaponType),(nkill)]])
outcome = classifier.predict(attack_params)
result = outcome[0]
outcome_result_dict = {
    0: 'Failure',
    1: 'Success'
}



print('The attack on Country Id: {} will be a {} based on the given parameters.'.format(country_id, outcome_result_dict[result]))
print('<br> With an Accuracy Score of: {}'.format(acc))


# In[ ]:

