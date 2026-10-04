-- Provider-level feature table for training AND Feature Online Store.
-- One row per provider_id. is_fraud is training-only.
-- Python substitutes: {project} {raw_dataset} {features_dataset}

CREATE OR REPLACE TABLE `{project}.{features_dataset}.provider_features` AS
WITH
claims AS (
  SELECT
    Provider AS provider_id,
    BeneID AS bene_id,
    ClaimID AS claim_id,
    ClaimStartDt AS claim_start,
    ClaimEndDt AS claim_end,
    InscClaimAmtReimbursed AS reimbursed,
    IFNULL(DeductibleAmtPaid, 0) AS deductible,
    AttendingPhysician AS attending_physician,
    OperatingPhysician AS operating_physician,
    ClmDiagnosisCode_1 AS diag_1,
    ClmDiagnosisCode_2 AS diag_2,
    ClmDiagnosisCode_3 AS diag_3,
    ClmDiagnosisCode_4 AS diag_4,
    ClmDiagnosisCode_5 AS diag_5,
    ClmProcedureCode_1 AS proc_1,
    ClmProcedureCode_2 AS proc_2,
    ClmProcedureCode_3 AS proc_3,
    CAST(NULL AS DATE) AS admission_dt,
    CAST(NULL AS DATE) AS discharge_dt,
    'outpatient' AS claim_type
  FROM `{project}.{raw_dataset}.outpatient`

  UNION ALL

  SELECT
    Provider AS provider_id,
    BeneID AS bene_id,
    ClaimID AS claim_id,
    ClaimStartDt AS claim_start,
    ClaimEndDt AS claim_end,
    InscClaimAmtReimbursed AS reimbursed,
    IFNULL(DeductibleAmtPaid, 0) AS deductible,
    AttendingPhysician AS attending_physician,
    OperatingPhysician AS operating_physician,
    ClmDiagnosisCode_1 AS diag_1,
    ClmDiagnosisCode_2 AS diag_2,
    ClmDiagnosisCode_3 AS diag_3,
    ClmDiagnosisCode_4 AS diag_4,
    ClmDiagnosisCode_5 AS diag_5,
    ClmProcedureCode_1 AS proc_1,
    ClmProcedureCode_2 AS proc_2,
    ClmProcedureCode_3 AS proc_3,
    AdmissionDt AS admission_dt,
    DischargeDt AS discharge_dt,
    'inpatient' AS claim_type
  FROM `{project}.{raw_dataset}.inpatient`
),

claim_enriched AS (
  SELECT
    c.*,
    DATE_DIFF(c.claim_end, c.claim_start, DAY) AS claim_duration_days,
    DATE_DIFF(c.discharge_dt, c.admission_dt, DAY) AS inpatient_los_days,
    DATE_DIFF(COALESCE(c.claim_start, CURRENT_DATE()), b.DOB, YEAR) AS bene_age,
    IF(b.ChronicCond_Alzheimer = 1, 1, 0) AS chron_alz,
    IF(b.ChronicCond_Heartfailure = 1, 1, 0) AS chron_hf,
    IF(b.ChronicCond_KidneyDisease = 1, 1, 0) AS chron_kidney,
    IF(b.ChronicCond_Cancer = 1, 1, 0) AS chron_cancer,
    IF(b.ChronicCond_Diabetes = 1, 1, 0) AS chron_diabetes,
    IFNULL(b.IPAnnualReimbursementAmt, 0) + IFNULL(b.OPAnnualReimbursementAmt, 0)
      AS bene_annual_reimb
  FROM claims c
  LEFT JOIN `{project}.{raw_dataset}.beneficiary` b
    ON c.bene_id = b.BeneID
),

diag_long AS (
  SELECT provider_id, claim_id, diag_code
  FROM claim_enriched
  UNPIVOT (diag_code FOR col IN (diag_1, diag_2, diag_3, diag_4, diag_5))
  WHERE diag_code IS NOT NULL AND diag_code != ''
),

proc_long AS (
  SELECT provider_id, claim_id, proc_code
  FROM claim_enriched
  UNPIVOT (proc_code FOR col IN (proc_1, proc_2, proc_3))
  WHERE proc_code IS NOT NULL AND proc_code != ''
),

diag_stats AS (
  SELECT
    provider_id,
    COUNT(DISTINCT diag_code) AS unique_diagnosis_codes
  FROM diag_long
  GROUP BY provider_id
),

proc_stats AS (
  SELECT
    provider_id,
    COUNT(DISTINCT proc_code) AS unique_procedure_codes
  FROM proc_long
  GROUP BY provider_id
),

top_diag AS (
  SELECT
    provider_id,
    diag_code,
    cnt,
    ROW_NUMBER() OVER (PARTITION BY provider_id ORDER BY cnt DESC, diag_code) AS rn
  FROM (
    SELECT provider_id, diag_code, COUNT(*) AS cnt
    FROM diag_long
    GROUP BY provider_id, diag_code
  )
),

provider_top_diag AS (
  SELECT provider_id, diag_code AS top_diag_code, cnt AS top_diag_cnt
  FROM top_diag
  WHERE rn = 1
),

provider_agg AS (
  SELECT
    provider_id,
    COUNT(*) AS claim_count,
    COUNTIF(claim_type = 'inpatient') AS inpatient_claim_count,
    COUNTIF(claim_type = 'outpatient') AS outpatient_claim_count,
    COUNT(DISTINCT bene_id) AS unique_beneficiary_count,
    SAFE_DIVIDE(COUNT(*), COUNT(DISTINCT bene_id)) AS claims_per_beneficiary,
    SUM(IFNULL(reimbursed, 0)) AS total_reimbursed,
    AVG(IFNULL(reimbursed, 0)) AS avg_reimbursed,
    MAX(IFNULL(reimbursed, 0)) AS max_reimbursed,
    IFNULL(STDDEV_SAMP(reimbursed), 0) AS std_reimbursed,
    SUM(IFNULL(deductible, 0)) AS total_deductible,
    SAFE_DIVIDE(
      SUM(IF(claim_type = 'inpatient', IFNULL(reimbursed, 0), 0)),
      NULLIF(SUM(IFNULL(reimbursed, 0)), 0)
    ) AS inpatient_reimb_share,
    SAFE_DIVIDE(
      SUM(IF(claim_type = 'outpatient', IFNULL(reimbursed, 0), 0)),
      NULLIF(SUM(IFNULL(reimbursed, 0)), 0)
    ) AS outpatient_reimb_share,
    AVG(claim_duration_days) AS avg_claim_duration_days,
    AVG(IF(claim_type = 'inpatient', inpatient_los_days, NULL)) AS avg_inpatient_los_days,
    COUNT(DISTINCT attending_physician) AS unique_attending_physicians,
    SAFE_DIVIDE(
      COUNTIF(operating_physician IS NOT NULL AND operating_physician != ''),
      COUNT(*)
    ) AS operating_physician_claim_share,
    AVG(bene_age) AS avg_beneficiary_age,
    AVG(chron_alz) AS chronic_alzheimers_share,
    AVG(chron_hf) AS chronic_heartfailure_share,
    AVG(chron_kidney) AS chronic_kidney_share,
    AVG(chron_cancer) AS chronic_cancer_share,
    AVG(chron_diabetes) AS chronic_diabetes_share,
    AVG(bene_annual_reimb) AS avg_annual_reimbursement
  FROM claim_enriched
  GROUP BY provider_id
)

SELECT
  a.provider_id,
  a.claim_count,
  a.inpatient_claim_count,
  a.outpatient_claim_count,
  a.unique_beneficiary_count,
  IFNULL(a.claims_per_beneficiary, 0) AS claims_per_beneficiary,
  a.total_reimbursed,
  a.avg_reimbursed,
  a.max_reimbursed,
  a.std_reimbursed,
  a.total_deductible,
  IFNULL(a.inpatient_reimb_share, 0) AS inpatient_reimb_share,
  IFNULL(a.outpatient_reimb_share, 0) AS outpatient_reimb_share,
  IFNULL(a.avg_claim_duration_days, 0) AS avg_claim_duration_days,
  IFNULL(a.avg_inpatient_los_days, 0) AS avg_inpatient_los_days,
  IFNULL(d.unique_diagnosis_codes, 0) AS unique_diagnosis_codes,
  IFNULL(p.unique_procedure_codes, 0) AS unique_procedure_codes,
  IFNULL(SAFE_DIVIDE(t.top_diag_cnt, a.claim_count), 0) AS top_diagnosis_share,
  a.unique_attending_physicians,
  IFNULL(a.operating_physician_claim_share, 0) AS operating_physician_claim_share,
  IFNULL(a.avg_beneficiary_age, 0) AS avg_beneficiary_age,
  IFNULL(a.chronic_alzheimers_share, 0) AS chronic_alzheimers_share,
  IFNULL(a.chronic_heartfailure_share, 0) AS chronic_heartfailure_share,
  IFNULL(a.chronic_kidney_share, 0) AS chronic_kidney_share,
  IFNULL(a.chronic_cancer_share, 0) AS chronic_cancer_share,
  IFNULL(a.chronic_diabetes_share, 0) AS chronic_diabetes_share,
  IFNULL(a.avg_annual_reimbursement, 0) AS avg_annual_reimbursement,
  CASE
    WHEN UPPER(l.PotentialFraud) = 'YES' THEN 1
    WHEN UPPER(l.PotentialFraud) = 'NO' THEN 0
    ELSE NULL
  END AS is_fraud
FROM provider_agg a
LEFT JOIN diag_stats d USING (provider_id)
LEFT JOIN proc_stats p USING (provider_id)
LEFT JOIN provider_top_diag t USING (provider_id)
LEFT JOIN `{project}.{raw_dataset}.labels` l
  ON a.provider_id = l.Provider
;
