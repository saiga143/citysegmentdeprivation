[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17642978.svg)](https://doi.org/10.5281/zenodo.17642978)

# City Segment Deprivation (CSD) Model — Analysis Workflow

This repository contains the **full processing, modelling, and comparative analysis pipeline** used to produce all results, figures, and tables for the city segment deprivation (CSD) study.  
The workflow includes:

- creation of labelled training data  
- Random Forest training & global application  
- comparison against SSI, Million Neighborhoods (MN), and WRI datasets  
- generation of manuscript figures and summary tables  

**Large datasets are *not* included** (e.g., rasters, country folders, city-level shapefiles), but all scripts required for reproduction are provided.

---

## 📁 Repository Structure

### **1. `1_preprocessing/`**
Prepares standardized city-segment data and benchmark-labelled training files.

- `01_preprocess_city_segments.ipynb`  
- `02_create_labeled_data.ipynb`  
- `LabelledData_for_RF/` — labelled CSVs used for RF training  

---

### **2. `2_modelling/`**

#### **2.1 `01_training/`**
VSURF variable selection and Random Forest training.

- `run_VSURF.R` — VSURF variable selection 
- `train_rf_model.py` — full RF training script  
- `rf_outputs/` — trained model artifacts  

#### **2.2 `02_application/`**
Application of the RF model to 5000+ cities.

- `01_analyse_rf_outputs.ipynb`  
- `02_apply_rf_predictions.ipynb`  
- `03_summary_statistics.ipynb`  
- `04_filtered_80_qc.ipynb`  
- `analysis_outputs/`  
- `predictions/`  
- `summary_statistics/`  

---

### **3. `3_comparitive_analysis/`**

#### **3.1 SSI**
- `01_SSI_DataRetrieval.js`  
- `02_ssi_clip_to_cities.ipynb`  
- `03_ssi_rf_comparison.ipynb`  
- `04_ssi_rf_comparison_figures_tables.ipynb`  
- `PerCountry_Outputs/`  
- `Pooled_Results/`  

#### **3.2 MN**
- `01_MN_Data_and_Labels.ipynb`  
- `02_MN_RF_comparison_Figures_tables.ipynb`  
- `Outputs/`  

#### **3.3 WRI**
- `01_WRI_DataDownload.js`  
- `02_WRI_IntersectionReports.ipynb`  
- `03_WRI_PerCountry_Metrics.ipynb`  
- `Outputs/`  
- `Intersect_reports/`  

---

### **4. `4_Figures_Tables/`**

Notebooks used to generate all manuscript figures and global summary tables.

#### **Figure Notebooks**
- `01_Figure2_Global_DeprivedShare.ipynb`  
- `02_Figure3_Lollipop_Citysize.ipynb`  
- `03_Figure4_Deprivation_by_Citysizemix.ipynb`  
- `04_Figure5_ThreeComparison.ipynb`  
- `05_GlobalSummaryTable.ipynb`  

#### **Supporting Input Files (small & included)**
- `AllCities_Points_new.gpkg`  
- `Country103_list_new.csv`  
- `country_ISOcodes_new.csv`  
- `world-administrative-boundaries-countries.shp`  
- `world_focus_Africa_Asia_LAC.gpkg`  

#### **Outputs**
- `Figures/` — manuscript figures  
- `Tables/` — cleaned summary tables  

---

## 📦 Notes on Large Files

Large datasets **not included** in the repo due to size and permission restrictions:

- SSI raster tiles  
- WRI LULC rasters (`PerCountry_Files`)  
- Global RF predictions (5000+ cities)  
- IDEABench, MN data (but redictred to the original source)

Each notebook includes a **"Paths to edit"** section for pointing to local folders.

---

## 🔄 Workflow Overview

1. **Preprocessing** → assemble training data  
2. **Modelling** → RF training + global application  
3. **Comparative Analysis** → SSI / MN / WRI alignment  
4. **Figures & Tables** → manuscript outputs  

All steps are modular and can be reproduced independently.

---

## 📘 Citation
If you use or build upon this work, please cite the paper and link to this repository.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17642978.svg)](https://doi.org/10.5281/zenodo.17642978)

Zenodo: Veeravalli, S. G. (2025). A Global, Standardized City Segment Deprivation (CSD) Model: Preprocessing, Training, Predictions, and Cross-Dataset Comparisons (v2.1). Zenodo. https://doi.org/10.5281/zenodo.17642978

Paper: _coming soon_

---

## 🙏 Acknowledgements
This work is supported by:
* FORMAS (Swedish Research Council for Sustainable Development), project DEPRIMAP (2023-01210) (https://sola.kau.se/deprimap/)
* The computation (model training) was partly enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725
* Thanks to CIESEN (for City Segments v1) and IDEAtlas (for IDEABench) datasets

<img width="373" height="110" alt="image" src="https://github.com/user-attachments/assets/a180a6e3-1b60-429d-b0b8-c14a45e4e190" />
