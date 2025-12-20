plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Distribusi Label Artikel')
plt.tight_layout()
plt.savefig('fig1_label_distribution.png')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['totalWordsCount'], bins=30, kde=True)
plt.title('Distribusi Jumlah Kata Artikel')
plt.tight_layout()
plt.savefig('fig2_total_words.png')
plt.show()


plt.figure(figsize=(12,10))
sns.heatmap(
    df.drop(columns=['TextID','URL','Label']).corr(),
    cmap='coolwarm',
    annot=False
)
plt.title('Heatmap Korelasi Fitur Numerik')
plt.tight_layout()
plt.savefig('fig3_correlation_heatmap.png')
plt.show()


