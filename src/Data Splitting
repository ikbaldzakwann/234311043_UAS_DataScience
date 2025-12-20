#TRAIN TEST SPLIT


TARGET = 'Label'
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)


X_train_numeric = X_train.drop(columns=['TextID', 'URL'])
X_test_numeric  = X_test.drop(columns=['TextID', 'URL'])

print(X_train_numeric.shape, X_test_numeric.shape)

# HAPUS KOLOM NON-FITUR SEBELUM TRAINING

X_train_numeric = X_train.drop(columns=['TextID', 'URL'])
X_test_numeric = X_test.drop(columns=['TextID', 'URL'])

