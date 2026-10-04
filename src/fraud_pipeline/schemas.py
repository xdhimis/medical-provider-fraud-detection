"""Shared schemas: raw table layouts, feature column contract, artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from google.cloud import bigquery

# Entity key used by Feature Online Store and prediction requests.
ENTITY_ID_COLUMN: Final[str] = "provider_id"
LABEL_COLUMN: Final[str] = "is_fraud"

# Columns written to the serving FeatureView / sent to the endpoint.
# Keep this list in sync with sql/provider_features.sql SELECT list
# (excluding provider_id and is_fraud).
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "claim_count",
    "inpatient_claim_count",
    "outpatient_claim_count",
    "unique_beneficiary_count",
    "claims_per_beneficiary",
    "total_reimbursed",
    "avg_reimbursed",
    "max_reimbursed",
    "std_reimbursed",
    "total_deductible",
    "inpatient_reimb_share",
    "outpatient_reimb_share",
    "avg_claim_duration_days",
    "avg_inpatient_los_days",
    "unique_diagnosis_codes",
    "unique_procedure_codes",
    "top_diagnosis_share",
    "unique_attending_physicians",
    "operating_physician_claim_share",
    "avg_beneficiary_age",
    "chronic_alzheimers_share",
    "chronic_heartfailure_share",
    "chronic_kidney_share",
    "chronic_cancer_share",
    "chronic_diabetes_share",
    "avg_annual_reimbursement",
)

RAW_TABLE_NAMES: Final[tuple[str, ...]] = (
    "labels",
    "beneficiary",
    "inpatient",
    "outpatient",
)

# Filename substrings under GCS_RAW_PREFIX -> BigQuery table name.
GCS_FILE_TO_TABLE: Final[dict[str, str]] = {
    "Train_Beneficiarydata": "beneficiary",
    "Train_Inpatientdata": "inpatient",
    "Train_Outpatientdata": "outpatient",
    # Labels file is named Train-<timestamp>.csv (no Beneficiary/Inpatient/etc.)
    "Train-": "labels",
}


def labels_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("Provider", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("PotentialFraud", "STRING", mode="REQUIRED"),
    ]


def beneficiary_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("BeneID", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("DOB", "DATE"),
        bigquery.SchemaField("DOD", "DATE"),
        bigquery.SchemaField("Gender", "INTEGER"),
        bigquery.SchemaField("Race", "INTEGER"),
        bigquery.SchemaField("RenalDiseaseIndicator", "STRING"),
        bigquery.SchemaField("State", "INTEGER"),
        bigquery.SchemaField("County", "INTEGER"),
        bigquery.SchemaField("NoOfMonths_PartACov", "INTEGER"),
        bigquery.SchemaField("NoOfMonths_PartBCov", "INTEGER"),
        bigquery.SchemaField("ChronicCond_Alzheimer", "INTEGER"),
        bigquery.SchemaField("ChronicCond_Heartfailure", "INTEGER"),
        bigquery.SchemaField("ChronicCond_KidneyDisease", "INTEGER"),
        bigquery.SchemaField("ChronicCond_Cancer", "INTEGER"),
        bigquery.SchemaField("ChronicCond_ObstrPulmonary", "INTEGER"),
        bigquery.SchemaField("ChronicCond_Depression", "INTEGER"),
        bigquery.SchemaField("ChronicCond_Diabetes", "INTEGER"),
        bigquery.SchemaField("ChronicCond_IschemicHeart", "INTEGER"),
        bigquery.SchemaField("ChronicCond_Osteoporasis", "INTEGER"),
        bigquery.SchemaField("ChronicCond_rheumatoidarthritis", "INTEGER"),
        bigquery.SchemaField("ChronicCond_stroke", "INTEGER"),
        bigquery.SchemaField("IPAnnualReimbursementAmt", "FLOAT"),
        bigquery.SchemaField("IPAnnualDeductibleAmt", "FLOAT"),
        bigquery.SchemaField("OPAnnualReimbursementAmt", "FLOAT"),
        bigquery.SchemaField("OPAnnualDeductibleAmt", "FLOAT"),
    ]


def inpatient_schema() -> list[bigquery.SchemaField]:
    # Order matches the Kaggle Train_Inpatientdata CSV headers.
    return [
        bigquery.SchemaField("BeneID", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ClaimID", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ClaimStartDt", "DATE"),
        bigquery.SchemaField("ClaimEndDt", "DATE"),
        bigquery.SchemaField("Provider", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("InscClaimAmtReimbursed", "FLOAT"),
        bigquery.SchemaField("AttendingPhysician", "STRING"),
        bigquery.SchemaField("OperatingPhysician", "STRING"),
        bigquery.SchemaField("OtherPhysician", "STRING"),
        bigquery.SchemaField("AdmissionDt", "DATE"),
        bigquery.SchemaField("ClmAdmitDiagnosisCode", "STRING"),
        bigquery.SchemaField("DeductibleAmtPaid", "FLOAT"),
        bigquery.SchemaField("DischargeDt", "DATE"),
        bigquery.SchemaField("DiagnosisGroupCode", "STRING"),
        *[bigquery.SchemaField(f"ClmDiagnosisCode_{i}", "STRING") for i in range(1, 11)],
        *[bigquery.SchemaField(f"ClmProcedureCode_{i}", "STRING") for i in range(1, 7)],
    ]


def outpatient_schema() -> list[bigquery.SchemaField]:
    # Order matches the Kaggle Train_Outpatientdata CSV headers
    # (diag/proc codes come before DeductibleAmtPaid / ClmAdmitDiagnosisCode).
    return [
        bigquery.SchemaField("BeneID", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ClaimID", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ClaimStartDt", "DATE"),
        bigquery.SchemaField("ClaimEndDt", "DATE"),
        bigquery.SchemaField("Provider", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("InscClaimAmtReimbursed", "FLOAT"),
        bigquery.SchemaField("AttendingPhysician", "STRING"),
        bigquery.SchemaField("OperatingPhysician", "STRING"),
        bigquery.SchemaField("OtherPhysician", "STRING"),
        *[bigquery.SchemaField(f"ClmDiagnosisCode_{i}", "STRING") for i in range(1, 11)],
        *[bigquery.SchemaField(f"ClmProcedureCode_{i}", "STRING") for i in range(1, 7)],
        bigquery.SchemaField("DeductibleAmtPaid", "FLOAT"),
        bigquery.SchemaField("ClmAdmitDiagnosisCode", "STRING"),
    ]


TABLE_SCHEMAS: Final[dict[str, list[bigquery.SchemaField]]] = {
    "labels": labels_schema(),
    "beneficiary": beneficiary_schema(),
    "inpatient": inpatient_schema(),
    "outpatient": outpatient_schema(),
}


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Paths written by the train step."""

    run_id: str
    gcs_uri: str
    model_filename: str = "model.bst"
    columns_filename: str = "feature_columns.json"
    metrics_filename: str = "metrics.json"
    threshold_filename: str = "threshold.json"

    @property
    def model_uri(self) -> str:
        return f"{self.gcs_uri}/{self.model_filename}"

    @property
    def columns_uri(self) -> str:
        return f"{self.gcs_uri}/{self.columns_filename}"

    @property
    def metrics_uri(self) -> str:
        return f"{self.gcs_uri}/{self.metrics_filename}"

    @property
    def threshold_uri(self) -> str:
        return f"{self.gcs_uri}/{self.threshold_filename}"
