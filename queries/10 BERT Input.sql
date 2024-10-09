CREATE OR REPLACE TABLE booming-edge-403620.mimic_thesis.bert_inputs AS
WITH VisitData AS (
    SELECT 
        person_id,
        STRING_AGG(
            CONCAT(
                '[DATE] ', CAST(visit_start_date AS STRING), 
                ' [DATE] ', CAST(visit_end_date AS STRING), 
                ' [CODES] ', (SELECT STRING_AGG(CAST(concept AS STRING), ' ') FROM UNNEST(concepts) concept),
                ' [SEP]'
            ), 
            ' '
        ) AS visit_info
    FROM 
        booming-edge-403620.mimic_thesis.final_patient_data,
        UNNEST(visits)
    GROUP BY 
        person_id
)

SELECT 
    p.person_id,
    CONCAT(
        '[DATE] ', CAST(p.year_of_birth AS STRING), 
        ' [DATE] ', IFNULL(CAST(p.death_date AS STRING), 'None'), 
        ' [CODES] ', IFNULL(CAST(p.death_cause_concept_id AS STRING), 'None'), 
        ' ', CAST(p.gender_concept_id AS STRING), 
        ' ', CAST(p.ethnicity_concept_id AS STRING),
        ' [SEP] ',
        COALESCE(v.visit_info, '')
    ) AS patient_text
FROM 
    booming-edge-403620.mimic_thesis.final_patient_data p
LEFT JOIN 
    VisitData v ON p.person_id = v.person_id;
