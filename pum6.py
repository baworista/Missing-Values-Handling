from sklearn.experimental import enable_iterative_imputer
import pandas as pd
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer

print("================================Task 1================================")

# Loading data from CSV file with appropriate delimiters
file_path = 'datasets/synthetic - tshirts.csv'
data = pd.read_csv(file_path, delimiter=';')

# Displaying the first few rows of data and information about missing values
print(data.head())
print(data.info())

# Filling missing values with mean (for numeric columns)
data_filled_mean = data.copy()
data_filled_mean['No of Items'] = data['No of Items'].fillna(data['No of Items'].mean())
data_filled_mean['Price'] = data['Price'].fillna(data['Price'].mean())

# Filling missing values with median (for numeric columns)
data_filled_median = data.copy()
data_filled_median['No of Items'] = data['No of Items'].fillna(data['No of Items'].median())
data_filled_median['Price'] = data['Price'].fillna(data['Price'].median())

# Filling missing values with the most frequent value (for categorical columns)
data_filled_mode = data.copy()
for column in data.select_dtypes(include=['object']).columns:
    data_filled_mode[column] = data[column].fillna(data[column].mode()[0])

# Filling missing values with forward fill
data_filled_ffill = data.ffill()

# Filling missing values with backward fill
data_filled_bfill = data.ffill()

# Displaying filled data
print("Data filled with mean:\n", data_filled_mean.head())
print("Data filled with median:\n", data_filled_median.head())
print("Data filled with mode:\n", data_filled_mode.head())
print("Data filled with forward fill:\n", data_filled_ffill.head())
print("Data filled with backward fill:\n", data_filled_bfill.head())


# Task 2, Airbnb dataset
print("================================Task 2================================")

# Loading data from CSV
file_path = 'datasets/Airbnb_Open_Data.csv'
df = pd.read_csv(file_path, delimiter=',', low_memory=False)

# Displaying the first few rows of data and information about missing values
print(df.info())

# Displaying the number of missing values in each column
print(df.isnull().sum())



# Removing 'license' column since it contains only 2 values out of 100k
df = df.drop(['license'],axis=1)

# Removing other unnecessary columns: 'country code', 'id', and 'host id'
df.drop(['NAME','country code', 'id', 'host id'], axis=1, inplace=True)

print("================================Data with deleted columns================================")
print(df.isnull().sum())
print("================================Data with deleted columns================================\n")



# Imputing missing values with median for 'reviews per month' column
df['reviews per month'] = df['reviews per month'].fillna(df['reviews per month'].median())

# Converting 'house_rules' column to category type
df['house_rules'] = df['house_rules'].astype('category')

# Adding a new category 'HR'
df['house_rules'] = df['house_rules'].cat.add_categories(['HR'])

# Filling missing values with the new category 'HR'
df['house_rules'] = df['house_rules'].fillna('HR')

print("================================Data filled with house_rules================================")
print(df.isnull().sum())
print("================================Data filled with house_rules================================\n")



# Forward filling missing values in 'review rate number' column
df['review rate number'] = df['review rate number'].ffill()
print("================================Data filled with forward fill================================")
print(df.isnull().sum())
print("================================Data filled with forward fill================================\n")



# Imputing missing values using KNN imputer
knn_imputer = KNNImputer(n_neighbors=5)
numeric_columns = df.select_dtypes(include=['float64']).columns
df[numeric_columns] = knn_imputer.fit_transform(df[numeric_columns])

print("================================Data filled with KNN imputation================================")
print(df.isnull().sum())
print("================================Data filled with KNN imputation================================\n")



# Dealing with date, no direct method
# Here we sort and then take the mean data between the previous and next

# Converting 'last review' column to datetime type
df['last review'] = pd.to_datetime(df['last review'])

# Sorting DataFrame by the 'last review' column
df.sort_values(by='last review', inplace=True)

# Finding the time interval between consecutive reviews
df['review_interval'] = df['last review'].diff().dt.days

# Imputing missing values in 'review_interval' with median
median_review_interval = df['review_interval'].median()
df['review_interval'] = df['review_interval'].fillna(median_review_interval)

# Calculating the missing 'last review' dates based on the existing ones and the imputed review intervals
df['last review'] = df['last review'].fillna(method='ffill') + pd.to_timedelta(df['review_interval'], unit='D')

print("================================Data with filled date================================")
print(df.isnull().sum())
print("================================Data with filled date================================\n")



# Iterative Imputer
# IterativeImputer in scikit-learn uses a regression model to estimate missing values based on other features in the dataset.

# Preprocessing 'price' and 'service fee' columns to remove non-numeric characters
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['service fee'] = df['service fee'].replace('[\$,]', '', regex=True).astype(float)

# Creating a DataFrame with only 'price' and 'service fee' columns
df_numeric = df[['price', 'service fee']]

# Initializing the IterativeImputer
model_imputer = IterativeImputer()

# Imputing missing values using the model-based imputer
df_imputed = pd.DataFrame(model_imputer.fit_transform(df_numeric), columns=df_numeric.columns)

# Replacing the imputed values in the original DataFrame
df[['price', 'service fee']] = df_imputed

print("================================Data filled with iterative imputation================================")
print(df.isnull().sum())
print("================================Data filled with iterative imputation================================\n")



# Simple Imputer
# Defining columns with categorical features
categorical_features = ['host name', 'host_identity_verified', 'country', 'instant_bookable']

# Initializing SimpleImputer with strategy as 'most_frequent'
imputer = SimpleImputer(strategy='most_frequent')

# Fitting the imputer on the DataFrame
imputer.fit(df[categorical_features])

# Transforming and replacing missing values in the DataFrame
df[categorical_features] = imputer.transform(df[categorical_features])

print("================================Data after simple imputer ================================")
print(df.isnull().sum())
print("================================Data after simple imputer ================================\n")



# After all, we have so few missing values that we can delete the rows which have at least one null value
df.dropna(axis=0, inplace=True)

print("================================Data after all ================================")
print(df.isnull().sum())
print("================================Data after all ================================\n")

