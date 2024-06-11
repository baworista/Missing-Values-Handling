from sklearn.experimental import enable_iterative_imputer
import pandas as pd
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer

print("================================Zadanie 1================================")

# Wczytanie danych z pliku CSV z odpowiednimi separatorami
file_path = 'datasets/synthetic - tshirts.csv'
data = pd.read_csv(file_path, delimiter=';')

# Wyświetlenie pierwszych kilku wierszy danych i informacji o brakujących wartościach
print(data.head())
print(data.info())

# Uzupełnianie braków średnią (dla kolumn numerycznych)
data_filled_mean = data.copy()
data_filled_mean['No of Items'] = data['No of Items'].fillna(data['No of Items'].mean())
data_filled_mean['Price'] = data['Price'].fillna(data['Price'].mean())

# Uzupełnianie braków medianą (dla kolumn numerycznych)
data_filled_median = data.copy()
data_filled_median['No of Items'] = data['No of Items'].fillna(data['No of Items'].median())
data_filled_median['Price'] = data['Price'].fillna(data['Price'].median())

# Uzupełnianie braków najczęstszą wartością (dla kolumn kategorycznych)
data_filled_mode = data.copy()
for column in data.select_dtypes(include=['object']).columns:
    data_filled_mode[column] = data[column].fillna(data[column].mode()[0])

# Uzupełnianie braków poprzednią dostępną wartością
data_filled_ffill = data.ffill()

# Uzupełnianie braków następną dostępną wartością
data_filled_bfill = data.ffill()

# Wyświetlenie uzupełnionych danych
print("Data filled with mean:\n", data_filled_mean.head())
print("Data filled with median:\n", data_filled_median.head())
print("Data filled with mode:\n", data_filled_mode.head())
print("Data filled with forward fill:\n", data_filled_ffill.head())
print("Data filled with backward fill:\n", data_filled_bfill.head())


# Zadanie 2, aibnb dataset
# https://medium.com/@pingsubhak/handling-missing-values-in-dataset-7-methods-that-you-need-to-know-5067d4e32b62
print("================================Zadanie 2================================")

# Wczytanie danych z pliku CSV z poprawnym separatorem (przecinek)
file_path = 'datasets/Airbnb_Open_Data.csv'
df = pd.read_csv(file_path, delimiter=',', low_memory=False)

# Wyświetlenie pierwszych kilku wierszy danych i informacji o brakujących wartościach
print(df.info())

# Wyświetlenie liczby brakujących wartości w każdej kolumnie
print(df.isnull().sum())



# Ponieważ z 100k w license mamy tylko 2 wartości, usunmy ta kolumne
df = df.drop(['license'],axis=1)

# Do tego usunmy niepotrzebne inne kolumny: 'country code', 'id', and 'host id' columns
df.drop(['NAME','country code', 'id', 'host id'], axis=1, inplace=True)

print("================================Data with deleted================================")
print(df.isnull().sum())



# Imputing missing values with mean/median of group
# Median
df['reviews per month'] = df['reviews per month'].fillna(df['reviews per month'].median())

# If we have many missing values in category, we can make new category for missing
# Konwertowanie kolumny 'house_rules' na typ 'category'
df['house_rules'] = df['house_rules'].astype('category')

# Dodawanie nowej kategorii 'H'
df['house_rules'] = df['house_rules'].cat.add_categories(['HR'])

# Wypełnianie brakujących wartości nową kategorią 'H'
df['house_rules'] = df['house_rules'].fillna('HR')

print("================================Data filled with house_rules================================")
print(df.isnull().sum())



# Forward fill missing values in a specific column
df['review rate number'] = df['review rate number'].ffill()

# Forward fill missing values in the entire DataFrame as example
ddff = df.copy()
ddff.ffill(inplace=True)
print("================================Data filled with forward fill================================")
print(ddff.isnull().sum())



# Impute missing values using KNN imputer
knn_imputer = KNNImputer(n_neighbors=5)
# Select only numeric columns for KNN imputation
numeric_columns = df.select_dtypes(include=['float64']).columns

# Apply KNN imputation to the numeric columns in the original DataFrame
df[numeric_columns] = knn_imputer.fit_transform(df[numeric_columns])

print("================================Data filled with KNN imputation================================")
print(df.isnull().sum())



# Dealing with date, no direct method
# Here we sort and than takin mean data between previous and next

# Convert 'last review' column to datetime type
df['last review'] = pd.to_datetime(df['last review'])

# Sort DataFrame by the 'last review' column
df.sort_values(by='last review', inplace=True)

# Find the time interval between consecutive reviews
df['review_interval'] = df['last review'].diff().dt.days

# Impute missing values in 'review_interval' with median
median_review_interval = df['review_interval'].median()
df['review_interval'] = df['review_interval'].fillna(median_review_interval)

# Calculate the missing 'last review' dates based on the existing ones and the imputed review intervals
df['last review'] = df['last review'].fillna(method='ffill') + pd.to_timedelta(df['review_interval'], unit='D')

print("================================Data filled with date================================")
print(df.isnull().sum())



# Iterative Imputer
# IterativeImputer in scikit-learn uses a regression model to estimate missing values based on other features in the dataset.
# Preprocess 'price' and 'service fee' columns to remove non-numeric characters
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['service fee'] = df['service fee'].replace('[\$,]', '', regex=True).astype(float)

# Create a DataFrame with only 'price' and 'service fee' columns
df_numeric = df[['price', 'service fee']]

# Initialize the IterativeImputer
model_imputer = IterativeImputer()

# Impute missing values using the model-based imputer
df_imputed = pd.DataFrame(model_imputer.fit_transform(df_numeric), columns=df_numeric.columns)

# Replace the imputed values in the original DataFrame
df[['price', 'service fee']] = df_imputed

# Check for missing values after imputation
print(df.isnull().sum())

print("================================Data filled with iterative imputation================================")
print(df.isnull().sum())



# Simple Imputer
# Define columns with categorical features
categorical_features = ['host name', 'host_identity_verified', 'country', 'instant_bookable']

# Initialize SimpleImputer with strategy as 'most_frequent'
imputer = SimpleImputer(strategy='most_frequent')

# Fit the imputer on your DataFrame
imputer.fit(df[categorical_features])

# Transform and replace missing values in the DataFrame
df[categorical_features] = imputer.transform(df[categorical_features])

print("================================Data after simple imputer ================================")
print(df.isnull().sum())



# And after all we have so small amount of missing values that we can delete
# the rows which has atleast one null value
df.dropna(axis=0, inplace=True)

print("================================Data after all ================================")
print(df.isnull().sum())

