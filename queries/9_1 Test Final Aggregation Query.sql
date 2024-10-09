CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.final_patient_data_test AS
SELECT 
    p.person_id,
    p.year_of_birth,
    p.gender_concept_id,
    p.ethnicity_concept_id,
    p.race_concept_id,
    d.death_date,
    d.cause_concept_id AS death_cause_concept_id,
    ARRAY_AGG(STRUCT(
        v.visit_occurrence_id,
        v.visit_start_date,
        v.visit_end_date,
        v.concepts
    )) AS visits
FROM 
    (SELECT * FROM booming-edge-403620.mimic_thesis.patient_base_data LIMIT 10) p
LEFT JOIN 
    booming-edge-403620.mimic_thesis.death_info d ON p.person_id = d.person_id
LEFT JOIN 
    booming-edge-403620.mimic_thesis.visit_aggregated_data v ON p.person_id = v.person_id
GROUP BY 
    p.person_id,
    p.year_of_birth,
    p.gender_concept_id,
    p.ethnicity_concept_id,
    p.race_concept_id,
    d.death_date,
    d.cause_concept_id;
