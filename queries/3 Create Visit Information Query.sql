CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.visit_info AS
SELECT 
    person_id,
    visit_occurrence_id,
    visit_start_date,
    visit_end_date
FROM 
    booming-edge-403620.mimiciv_full_current_cdm.visit_occurrence;
