CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.visit_observations AS
SELECT 
    visit_occurrence_id,
    ARRAY_AGG(DISTINCT observation_concept_id) AS observation_concepts
FROM 
    booming-edge-403620.mimiciv_full_current_cdm.observation
GROUP BY 
    visit_occurrence_id;
