CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.visit_procedures AS
SELECT 
    visit_occurrence_id,
    ARRAY_AGG(DISTINCT procedure_concept_id) AS procedure_concepts
FROM 
    booming-edge-403620.mimiciv_full_current_cdm.procedure_occurrence
GROUP BY 
    visit_occurrence_id;
