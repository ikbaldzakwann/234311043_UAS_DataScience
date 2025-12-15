# 📘 Judul Proyek
*KLASIFIKASI OBJEKTIVITAS ARTIKEL OLAHRAGA MENGGUNAKAN LOGISTIC REGRESSION, RANDOM FOREST DAN MULTILAYER PERCEPTRON*

## 👤 Informasi
- **Nama:** Ikbal Dzakwan  
- **Repo:** https://github.com/ikbaldzakwann/234311043_UAS_DataScience  
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
- Melakukan data preparation untuk memastikan kualitas dataset
- Membangun tiga model klasifikasi (Baseline, Advanced ML, dan Deep Learning)
- Melakukan evaluasi model menggunakan metrik accuracy, precision, recall, dan F1-score
- Membandingkan performa model dan menentukan model terbaik

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- [ Diperlukan model klasifikasi untuk membedakan artikel olahraga objektif dan subjektif secara otomatis.]
- [ Data teks memiliki pola yang kompleks dan non-linear sehingga model linear sederhana belum tentu optimal.]
- [ Dataset memiliki banyak fitur numerik hasil ekstraksi teks sehingga diperlukan preprocessing untuk menghindari noise dan redundansi.]
- [ Diperlukan perbandingan performa antara model baseline, machine learning, dan deep learning.]  

**Goals:**  
- [Membangun sistem klasifikasi objektivitas artikel olahraga menggunakan Logistic Regression, Random Forest, dan MLP.]  
- [Membandingkan performa ketiga model menggunakan metrik evaluasi klasifikasi.]
- [Menentukan model terbaik berdasarkan performa data uji.]
- [Menyediakan pipeline analisis yang reproducible.]  

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_cnn.h5
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** [ UCI Machine Learning Repository – Sports Articles for Objectivity Analysis]
- **Jumlah Data:** [Jumlah Data: 1994 baris, 127 fitur numerik, dan 1 target ]
- **Tipe:** [Tabular (numerik)]  

### Fitur Utama
Dataset memiliki 127 fitur numerik hasil ekstraksi teks.
Pada proyek ini digunakan 10 fitur numerik utama yang paling relevan.
| Fitur           | Deskripsi                       |
| --------------- | ------------------------------- |
| totalWordsCount | Jumlah kata dalam artikel       |
| numSentences    | Jumlah kalimat                  |
| avgWordLength   | Rata-rata panjang kata          |
| numNouns        | Jumlah kata benda               |
| numVerbs        | Jumlah kata kerja               |
| numAdjectives   | Jumlah kata sifat               |
| numAdverbs      | Jumlah kata keterangan          |
| numPronouns     | Jumlah kata ganti               |
| polarityScore   | Skor polaritas sentimen         |
| label           | Target (objective / subjective) |


---

# 4. 🔧 Data Preparation
- Cleaning: Pemeriksaan missing values dan duplikasi data 
- Feature Selection: Menggunakan 10 fitur numerik utama 
- Transformasi:
    -StandardScaler untuk Logistic Regression dan MLP  
-Splitting:
  - 80% training
  - 20% testing (stratified)
---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** [Logistic Regression]  
- **Model 2 – Advanced ML:** [Random Forest]  
- **Model 3 – Deep Learning:** [Multilayer Perceptron (MLP)]  

---

# 6. 🧪 Evaluation
**Metrik evaluasi:** 
Accuracy, Precision, Recall, dan F1-score
### Hasil Singkat
| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.85     | 0.85      | 0.85   | 0.85     |
| Random Forest       | 0.83     | 0.83      | 0.83   | 0.83     |
| MLP                 | 0.88     | 0.87      | 0.86   | 0.87     |


---

# 7. 🏁 Kesimpulan
- Model terbaik: [MLP (Deep Learning)]  
- Alasan: -Mampu menangkap pola non-linear yang lebih kompleks
           -Memberikan performa evaluasi tertinggi  
- Insight penting: - Fitur linguistik seperti jumlah kata dan polaritas memiliki pengaruh besar
                   - Random Forest unggul dibanding baseline karena mampu menangani hubungan non-linear  

---

# 8. 🔮 Future Work
- [ ] Tambah data  
- [ ] Tuning model  
- [ ] Coba arsitektur DL lain  
- [ ] Deployment  

---

# 9. 🔁 Reproducibility
Gunakan environment:
- Python: 3.12
Libraries utama:
  - numpy==1.26.4
  - pandas==2.2.2
  - scikit-learn==1.5.1
  - matplotlib==3.7.1
  - seaborn==0.13.2
  - tensorflow==2.15.0
Random seed digunakan untuk memastikan hasil eksperimen dapat direproduksi.
