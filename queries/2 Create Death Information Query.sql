CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.death_info AS
SELECT 
    person_id,
    MIN(death_date) AS death_date,  -- Assuming one death date per person
    cause_concept_id
FROM 
    booming-edge-403620.mimiciv_full_current_cdm.death
GROUP BY 
    person_id, 
    cause_concept_id;
