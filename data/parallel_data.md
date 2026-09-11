
It contains **team-authored agricultural Q&A pairs** covering Sri Lanka and South Africa farming contexts.  
This dataset supplements the official Kaggle Agriculture & Climate SLM dataset and demonstrates our ability to create domain‑specific, values‑aligned agricultural data.

---

## 1. Purpose of the Team-Generated Dataset

Our team had already been working on a **Sri Lanka + South Africa Agricultural SLM project** before the TRI challenge began.  
During that work, we created several synthetic Q&A datasets to explore crop issues, climate adaptation, and extension-style guidance across both countries.

To support the TRI educational objectives, we **aligned our existing synthetic datasets to the Kaggle Agriculture & Climate SLM benchmark**.  
This alignment ensures:

- consistency with Kaggle’s Q&A structure  
- compatibility with TRI’s evaluation format  
- clear separation between Kaggle-provided data and team-generated data  
- transparent documentation of our original project work  

Although this dataset originated from our independent AgriSLM project, it has been **refined and adapted specifically for TRI educational use**.

This dataset is intended for:

- model experimentation  
- domain adaptation  
- synthetic data augmentation  
- demonstration of team capability  
- TRI educational use  

---

## 2. Dataset File

**Filename:** `agrislm_team_demo_dataset.csv`  
**Type:** Synthetic Q&A dataset  
**Format:** CSV  
**Rows:** 
**Columns:** 

These columns allow the dataset to capture both **agronomic content** and **metadata** for fine-grained model conditioning.

---

## 3. Column Description

| Column         | Description |
|----------------|-------------|
| **question**   | Farmer-style query written by the team |
| **answer**     | Concise extension-style response |
| **region**     | `sri_lanka` or `south_africa` |
| **crop**       | Crop or livestock category |
| **agro_zone**  | Highland, semi-arid, sub-humid, etc. |
| **category**   | Disease, nutrient, climate, livestock, etc. |
| **climate_event** | Rain, drought, frost, heat stress |
| **safety_flag** | `chemical_use`, `none`, or `warning` |

This schema aligns with TRI’s values-led safety guidelines.

---

## 4. Sample Entries 

### **Example 1 — Sri Lanka (Highland Zone)**
**Q:** Heavy rains are causing leaf blight in my potato field.  
**A:** Improve drainage, remove infected leaves, and apply a copper fungicide safely.  
**Metadata:** `sri_lanka`, `potato`, `highland`, `disease`, `heavy_rain`, `chemical_use`

---

### **Example 2 — South Africa (Semi-Arid Zone)**
**Q:** Grass is disappearing earlier this season — what feed strategy should I use?  
**A:** Use conserved hay, drought-tolerant fodder, and avoid sudden ration changes.  
**Metadata:** `south_africa`, `livestock`, `semi_arid`, `livestock`, `drought`, `none`

---

### **Example 3 — Sri Lanka (Beans)**
**Q:** My bean leaves have small reddish spots during the rains.  
**A:** Likely bean rust — remove affected leaves and improve airflow.  
**Metadata:** `sri_lanka`, `beans`, `highland`, `disease`, `rain`, `none`

---

## 5. Ethical & Safety Notes

- All entries are **synthetic** and manually authored by Team QueenElizabeth.  
- No scraped, copyrighted, or personal farmer data is included.  
- Chemical advice follows DoA Sri Lanka and DALRRD South Africa guidelines.  
- Safety flags highlight entries involving chemical use or risk.  
- Dataset is for TRI educational and experimental use only.

---

## 6. How This Dataset Supports AgriSLM

This dataset strengthens AgriSLM by:

- adding localized Sri Lanka + South Africa agricultural knowledge  
- improving climate adaptation coverage  
- providing realistic extension-style Q&A  
- enabling domain-specific fine-tuning  
- demonstrating team capability in synthetic data creation  

It complements the Kaggle dataset and aligns with TRI’s values-led problem framing.

---

## 7. License

This synthetic dataset is released under **CC BY-SA 4.0**, allowing reuse with attribution and share-alike conditions.

---

# End of parallel_data.md

