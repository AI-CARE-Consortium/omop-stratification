CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.patient_base_data AS
SELECT 
    person_id,
    year_of_birth,
    gender_concept_id,
    ethnicity_concept_id,
    race_concept_id
FROM 
    booming-edge-403620.mimiciv_full_current_cdm.person;
