# 🎬 User-Based KNN Recommender System with Custom NDCG Evaluation

Bu proje, **MovieLens 100k** veri seti kullanılarak oluşturulmuş **User-Based K-Nearset Neighbors (KNN)** tabanlı bir tavsiye sistemi içerir.  
Model, `scikit-surprise` kütüphanesi ile geliştirilmiş olup özel bir **NDCG@10** değerlendirme sınıfına sahiptir.

## 📋 Project Overview

Bu çalışmanın amacı, kullanıcıların film puanlamalarına dayanarak **kişiye özel film önerileri** üretmektir.  
Model, hem **hata metrikleri** hem de **sıralama kalitesi** üzerinden değerlendirilmiştir.

### 🔍 Key Features
- Algoritma: `KNNWithMeans` (User-Based)
- Benzerlik Ölçütü: Pearson Correlation
- Doğrulama: 5-Fold Cross Validation
- Komşu Sayısı (k): 50
- Filtreleme: Top-N önerilerde tahmin puanı > 3.5
- Özel Metod: Custom `NDCGEvaluator` sınıfı ile NDCG@10 hesabı

## 🛠️ Installation & Requirements

Projeyi çalıştırmak için Python ve `scikit-surprise` kurulu olmalıdır.

### Repository’yi Klonla
```bash
git clone https://github.com/SevdaErgun/MovieLens-Recommender.git
cd MovieLens-Recommender
```

### Gerekli Kütüphaneleri Kur
```bash
pip install scikit-surprise numpy
```

## 🚀 Usage

Aşağıdaki komutla projeyi başlatabilirsiniz:

```bash
python main.py
```

Kod MovieLens 100k veri setini otomatik indirir ve 5-fold doğrulama sürecini yürütür.

## 📊 Methodology & Steps

Proje aşağıdaki adımları izler:

1. Data Loading  
2. Model Training  
3. Prediction  
4. Evaluation (MAE)  
5. Top-10 Recommendation Generation  
6. Precision & Recall Calculation  
7. NDCG Calculation  

## 📈 Results

| Fold | MAE | Precision@10 | Recall@10 | NDCG@10 |
|------|------|--------------|-----------|---------|
| Fold 1 | 0.7405 | 0.6759 | 0.5254 | 0.9162 |
| Fold 2 | 0.7393 | 0.6954 | 0.5305 | 0.9193 |
| Fold 3 | 0.7479 | 0.6876 | 0.5283 | 0.9166 |
| Fold 4 | 0.7426 | 0.6885 | 0.5253 | 0.9160 |
| Fold 5 | 0.7419 | 0.6941 | 0.5254 | 0.9180 |
| Average | 0.7424 | 0.6883 | 0.5270 | 0.9172 |
