# Team-Generated Parallel Data (Synthetic Q&A)

This file documents the supplementary **synthetic parallel dataset** created by Team QueenElizabeth to complement the official Kaggle *Agriculture & Climate SLM* dataset.  
The purpose of this dataset is to enrich low-resource agricultural scenarios from **Sri Lanka and South Africa**, especially where official extension documents are limited.

---

## 1. Purpose of the Parallel Dataset

Smallholder farmers in Sri Lanka and South Africa face localized challenges that are not fully covered in the Kaggle dataset.  
To address this gap, our team manually authored additional Q&A pairs that:

- reflect region-specific agricultural issues  
- incorporate climate adaptation scenarios  
- follow TRI’s values-led safety guidelines  
- remain fully synthetic (no scraping, no copyrighted content, no personal data)

This dataset is intended for **educational and experimental use** within the TRI challenge.

---

## 2. Dataset Description

- **Total entries:** X Q&A pairs (Sri Lanka + South Africa)  
- **Type:** Fully synthetic, manually authored  
- **Format:** Simple Q&A pairs with optional metadata  
- **Domains covered:**
  - Crop diseases (beans, potato, maize)
  - Soil and nutrient management
  - Climate adaptation (rainfall, drought, frost)
  - Livestock feeding strategies
  - Farmer safety and chemical-use warnings

---

## 3. Sample Entries

Below are a few representative examples from the synthetic dataset:

### **Example 1 — Sri Lanka (Highland Zone)**
**Q:** Heavy rains are causing leaf blight in my potato field. What should I do?  
**A:** Improve drainage, avoid overhead irrigation, remove infected leaves, and apply a copper-based fungicide following DoA guidelines.

---

### **Example 2 — South Africa (Semi-Arid Zone)**
**Q:** Grass is disappearing earlier this season. How can I adjust my livestock feed?  
**A:** Introduce conserved hay, supplement with drought-tolerant fodder, and avoid sudden ration changes to reduce stress.

---

### **Example 3 — Sri Lanka (Vegetable Crop)**
**Q:** My beans have small reddish spots during the rainy season.  
**A:** Likely bean rust — remove affected leaves, improve airflow, and apply a safe fungicide if needed.

---

## 4. Metadata Schema (Optional)

If metadata is used, entries follow this structure:

```json
{
  "question": "...",
  "answer": "...",
  "region": "sri_lanka | south_africa",
  "crop": "potato | maize | beans | livestock",
  "agro_zone": "highland | semi_arid | sub_humid",
  "climate_event": "heavy_rain | drought | frost",
  "safety_flag": "chemical_use | none"
}
