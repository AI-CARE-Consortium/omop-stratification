# Enhancing Stratification for Survival Analyses<br>across Standardized Data Sources
## Generation of Patient Embeddings using BERT 
### Structure of Repository
- **clustering_cancer_registry.xlsx**: clustering analysis for cancer registry data;
- **clustering_mimic.xlsx**: clustering analysis for MIMIC;
- **embeddings/**:
    - **config.yaml**: configuration file for all the scripts;
    - **tokenizer.py**: tokenizer class and additional methods;
    - **training.py**: BERT pre-training script;
    - **inference.py**: inference script for BERT (embeddings generation);
    - **clustering.py**: clustering and analysis;
    - **load_cancer_data.py**: loads and prepares cancer registry data, generates patient sequences, creates labels;
    - **create_mimic_labels.py**: prepares MIMIC data, creates labels;
- **queries/**:  SQL queries used in Google Cloud for creating a patient sequences for MIMIC data;

## Cluster Evaluation with Survival Analysis
For the evaluation of clusters using survival analysis, we utilized the code provided by Germer et al. 
> Germer, S., Rudolph, C., Laboohm, L., Katalinic, A., Rath, N., Rausch, K., Holleczek, B., the AI-CARE Working Group & Handels, H. (2024). **Survival analysis for lung cancer patients: A comparison of Cox regression and machine learning models**. International Journal of Medical Informatics, 191.  
> [https://doi.org/10.1016/j.ijmedinf.2024.105607](https://doi.org/10.1016/j.ijmedinf.2024.105607)

The original code can be found in the following repository:  
[https://github.com/AI-CARE-Consortium/survival-analysis-lung-cancer](https://github.com/AI-CARE-Consortium/survival-analysis-lung-cancer)

## How to Cite
A journal paper is currently under review.

> Shubov, M., Beernink, M., AI-CARE-Consortium, Gundler, C. (2024). **Enhancing Stratification for Survival Analyses across Standardized Data Sources**.
<!-- > [https://doi.org/10.5281/zenodo.10377082](https://doi.org/10.5281/zenodo.10377082) -->
