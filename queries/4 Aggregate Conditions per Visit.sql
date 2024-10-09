CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.visit_conditions AS
SELECT 
    visit_occurrence_id,
    ARRAY_AGG(DISTINCT condition_concept_id) AS condition_concepts
FROM 
    booming-edge-403620.mimiciv_full_current_cdm.condition_occurrence
GROUP BY 
    visit_occurrence_id;
