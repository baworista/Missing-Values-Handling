Here's a list of all the methods used in the code:

**Drop columns**: Removed the 'license', 'country code', 'id', 'host id', and 'NAME' columns from the DataFrame.

**Fill missing values with median**: Imputed missing values in the 'reviews per month' column with the median value of that column.

**Create a new category for missing values**: Converted the 'house_rules' column to categorical type and filled missing values with a new category 'HR'.

**Forward fill**: Filled missing values in the 'review rate number' column using forward fill method.

**KNN imputation**: Imputed missing values using KNN imputer with 5 nearest neighbors.

**Imputation based on time intervals**: Sorted DataFrame by 'last review' column, calculated time intervals between consecutive reviews, and filled missing values in 'review_interval' column with median. Then imputed missing 'last review' dates based on existing dates and imputed review intervals.

**Iterative imputation**: Used IterativeImputer from scikit-learn to estimate missing values based on other features. Preprocessed 'price' and 'service fee' columns to remove non-numeric characters before imputation.

**Simple imputer**: Imputed missing values in categorical features ('host name', 'host_identity_verified', 'country', 'instant_bookable') using SimpleImputer with 'most_frequent' strategy.

**Drop rows with missing values**: Finally, dropped rows with at least one null value from the DataFrame.
