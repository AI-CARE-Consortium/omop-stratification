CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.visit_drugs AS
SELECT 
    visit_occurrence_id,
    ARRAY_AGG(DISTINCT drug_concept_id) AS drug_concepts
FROM 
    booming-edge-403620.mimiciv_full_current_cdm.drug_exposure
GROUP BY 
    visit_occurrence_id;
