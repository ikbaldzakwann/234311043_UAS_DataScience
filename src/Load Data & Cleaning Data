# LOAD DATA

df = pd.read_csv('/content/Data1.csv')
df.head()

# DATA CLEANING

# Drop duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Imputasi missing values hanya untuk kolom numerik
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print("Jumlah duplikasi:", df.duplicated().sum())
print("Jumlah missing value:\n", df.isnull().sum())
