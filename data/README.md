# data/


## Description

| File | Description | Size |
|---|---|---|
| `Fake.csv` | ISOT fake news articles (cross-domain evaluation) | ~62MB |
| `True.csv` | ISOT real news articles (cross-domain evaluation) | ~53MB |

---


| File | Description | Source |
|---|---|---|
| `Truth_Seeker_Model_Dataset.csv` | Primary training dataset — 134,194 labeled tweets (2009–2022) | https://ieee-dataport.org/documents/truthseeker |

---

## 📥 Dataset Downloads

### ISOT Fake & True News Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Contains:
- `Fake.csv` → Fake news articles  
- `True.csv` → Real news articles  

---

### TruthSeeker-2023 Dataset
1. Visit: https://ieee-dataport.org/documents/truthseeker  
2. Download: `Truth_Seeker_Model_Dataset.csv`

---

## ⚙️ Expected Format

The script expects:

- `tweet` → text content  
- `BinaryNumTarget` → label (0 = fake, 1 = real)