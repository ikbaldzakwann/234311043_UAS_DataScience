#Perbandingan 3 Model 

results = pd.DataFrame({
    'Accuracy': [lr_acc, rf_acc, dl_acc],
    'Precision': [lr_prec, rf_prec, dl_prec],
    'Recall': [lr_rec, rf_rec, dl_rec],
    'F1-Score': [lr_f1, rf_f1, dl_f1],
    'Training Time (s)': [lr_time, rf_time, mlp_time]
}, index=['Logistic Regression', 'Random Forest', 'MLP (Deep Learning)'])

results

# PERBANDINGAN WAKTU TRAINING


plt.figure(figsize=(8,5))
plt.bar(
    results.index,
    results['Training Time (s)']
)

plt.title('Perbandingan Waktu Training Model')
plt.ylabel('Detik')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig('comparison_training_time.png')
plt.show()


# ===============================
# MODEL TERBAIK (F1-SCORE)
# ===============================

plt.figure(figsize=(8,5))
sns.barplot(
    x=results.index,
    y=results['F1-Score']
)

plt.title('Perbandingan Model Berdasarkan F1-Score')
plt.ylabel('F1-Score')
plt.xlabel('Model')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('comparison_best_model.png')
plt.show()


# ===============================
# PERBANDINGAN 3 MODEL (METRIK)
# ===============================

results[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar',
    figsize=(10,6)
)

plt.title('Perbandingan Performa 3 Model Klasifikasi')
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('comparison_metrics_models.png')
plt.show()


plt.figure(figsize=(10,6))
results.plot(kind='bar')
plt.title('Perbandingan Performa 3 Model')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('comparison_models.png')
plt.show()


mlp_model.summary()

mlp_model.save('mlp_model.keras')
mlp_model.save('mlp_model.h5')
