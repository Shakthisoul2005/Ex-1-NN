<H3> SHAKTHI VEL V </H3>
<H3> 212224240149 </H3>
<H3> EX. NO.1 </H3>
<H3> 21/4/2026 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```py
#import libraries

from google.colab import files
import pandas as pd
import io 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
```py
#Read the dataset from drive
df = pd.read_csv('/content/Churn_Modelling.csv');
print(df)
```
```py
#split the dataset
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:, -1].values
print(y)
```
```py
#Finding Missing Values
print(df.isnull().sum())
```
```py
#Handling Missing values
df.fillna(df.select_dtypes(include='number').mean().round(1), inplace=True)
print(df.isnull().sum())
y = df.iloc[:, -1].values
print(y)
```
```py
df.drop(['Surname','Geography','Gender'], axis=1, inplace = True)
df.info()
```
```py
#Check for Duplicates
df.duplicated()
```
```py
#Detect Outliers
print(df['EstimatedSalary'].describe())
```
```py
#When we normalize the dataset it brings the value of all the features
#between 0 and 1 so that all the columns re in the same range,
#and thus there is no dominant feature.

scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)
```
```py
#splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

#'test_size = 0.2' means 20% test data and 80% train data
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
```


## OUTPUT:

<img width="1490" height="673" alt="image" src="https://github.com/user-attachments/assets/dbc517e3-c776-43c9-9361-2bdd96bd2c8b" />

<img width="1075" height="331" alt="image" src="https://github.com/user-attachments/assets/b3657bf8-5a4f-46a8-8561-025b54b2e630" />

<img width="1295" height="415" alt="image" src="https://github.com/user-attachments/assets/32ae90ee-bc49-425a-a88b-d59461dc5b8f" />

<img width="1183" height="514" alt="image" src="https://github.com/user-attachments/assets/518a859c-03f5-4da4-8602-2a70e1cc2718" />

<img width="1257" height="473" alt="image" src="https://github.com/user-attachments/assets/6c201762-f6b7-448e-ad4a-fe4f16290d3a" />

<img width="1082" height="640" alt="image" src="https://github.com/user-attachments/assets/ecbe117e-92ab-43e9-9c1f-6452acc347b3" />

<img width="1227" height="288" alt="image" src="https://github.com/user-attachments/assets/1097255a-6b53-403f-b015-0a67913320be" />

<img width="1243" height="509" alt="image" src="https://github.com/user-attachments/assets/07229ef7-688e-4475-b348-a4d3a3ab4bd1" />

<img width="1335" height="588" alt="image" src="https://github.com/user-attachments/assets/3ee0522b-b06a-4615-b863-33d1c2d7816a" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


