
# MODEL 1: LOGISTIC REGRESSION

start = time.time()
lr_model = Pipeline(
    [('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=SEED))]
)
lr_model.fit(X_train_numeric, y_train)
lr_time = time.time() - start
print(f"Training Time Logistic Regression: {lr_time:.4f} detik")

# Prediction & Evaluation
y_pred_lr = lr_model.predict(X_test_numeric)

lr_acc  = accuracy_score(y_test, y_pred_lr)
lr_prec = precision_score(y_test, y_pred_lr, average='weighted')
lr_rec  = recall_score(y_test, y_pred_lr, average='weighted')
lr_f1   = f1_score(y_test, y_pred_lr, average='weighted')

print("=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, y_pred_lr))

# Overfitting Check
train_acc_lr = lr_model.score(X_train_numeric, y_train)
test_acc_lr  = lr_model.score(X_test_numeric, y_test)
print(f"Train Accuracy : {train_acc_lr:.4f}")
print(f"Test Accuracy  : {test_acc_lr:.4f}")
print(f"Gap            : {train_acc_lr - test_acc_lr:.4f}")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=lr_model.named_steps['lr'].classes_,
    yticklabels=lr_model.named_steps['lr'].classes_
)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig('cm_logistic_regression.png') # Added savefig
plt.show()

# Save model
joblib.dump(lr_model, 'lr_model.joblib')
print("Logistic Regression model saved as 'lr_model.joblib'")

# MODEL 2: RANDOM FOREST

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=0.5,
    random_state=SEED,
    n_jobs=-1
)
start = time.time()
rf_model.fit(X_train_numeric, y_train)
rf_time = time.time() - start
print(f"Training Time Random Forest: {rf_time:.4f} detik")

# Prediction & Evaluation
y_pred_rf = rf_model.predict(X_test_numeric)
rf_acc  = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf, average='weighted')
rf_rec  = recall_score(y_test, y_pred_rf, average='weighted')
rf_f1   = f1_score(y_test, y_pred_rf, average='weighted')

print("=== RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf))

# Overfitting Check
train_acc_rf = rf_model.score(X_train_numeric, y_train)
test_acc_rf  = rf_model.score(X_test_numeric, y_test)
print(f"Train Accuracy : {train_acc_rf:.4f}")
print(f"Test Accuracy  : {test_acc_rf:.4f}")
print(f"Gap            : {train_acc_rf - test_acc_rf:.4f}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig('cm_random_forest.png') # Added savefig
plt.show()

# Save model
joblib.dump(rf_model, 'rf_model.joblib')
print("Random Forest model saved as 'rf_model.joblib'")

# Split data latih menjadi train & validation (10% dari data latih)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_numeric,
    y_train,
    test_size=0.1,
    random_state=SEED,
    stratify=y_train
)

# Scaling fitur (fit HANYA di data train)
scaler = StandardScaler()
X_tr_scaled  = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_numeric)

# Label encoding
le = LabelEncoder()
y_tr_enc  = le.fit_transform(y_tr)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# Model MLP
mlp_model = Sequential([
    Input(shape=(X_tr_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training
start = time.time()
history = mlp_model.fit(
    X_tr_scaled,
    y_tr_enc,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_scaled, y_val_enc),
    verbose=1
)
mlp_time = time.time() - start
print("Training Time MLP:", round(mlp_time, 4), "detik")

# Evaluation
y_pred_mlp = (mlp_model.predict(X_test_scaled) > 0.5).astype(int).ravel()

dl_acc  = accuracy_score(y_test_enc, y_pred_mlp)
dl_prec = precision_score(y_test_enc, y_pred_mlp)
dl_rec  = recall_score(y_test_enc, y_pred_mlp)
dl_f1   = f1_score(y_test_enc, y_pred_mlp)


print("=== MLP ===")
print(classification_report(y_test_enc, y_pred_mlp))

# Save model
from keras.saving import save_model
save_model(mlp_model, "mlp_model.keras")
print("MLP model saved as 'mlp_model.keras'")

mlp_model.save('mlp_model.h5')
print("MLP model saved as 'mlp_model.h5'")

# Plot training history
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('MLP Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('mlp_accuracy.png') # Added savefig
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('MLP Loss')
plt.legend()
plt.tight_layout()
plt.savefig('mlp_loss.png') # Added savefig
plt.show()
