CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.visit_aggregated_data AS
SELECT 
    v.person_id,
    v.visit_occurrence_id,
    v.visit_start_date,
    v.visit_end_date,
    ARRAY_CONCAT(
        COALESCE(c.condition_concepts, []),
        COALESCE(p.procedure_concepts, []),
        COALESCE(d.drug_concepts, []),
        COALESCE(o.observation_concepts, [])
    ) AS concepts
FROM 
    booming-edge-403620.mimic_thesis.visit_info v
LEFT JOIN 
    booming-edge-403620.mimic_thesis.visit_conditions c ON v.visit_occurrence_id = c.visit_occurrence_id
LEFT JOIN 
    booming-edge-403620.mimic_thesis.visit_procedures p ON v.visit_occurrence_id = p.visit_occurrence_id
LEFT JOIN 
    booming-edge-403620.mimic_thesis.visit_drugs d ON v.visit_occurrence_id = d.visit_occurrence_id
LEFT JOIN 
    booming-edge-403620.mimic_thesis.visit_observations o ON v.visit_occurrence_id = o.visit_occurrence_id;
