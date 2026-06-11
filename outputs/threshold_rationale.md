# Threshold Optimisation Rationale

## Clinical Context

In a hospital readmission reduction programme, there are two types of
errors:

1. **False Negative (missed readmission)** — A patient who *will* be
   readmitted is not flagged.  This is costly: the hospital misses an
   opportunity for transitional care (discharge planning, medication
   reconciliation, home-health referral) and may incur CMS penalties.

2. **False Positive (unnecessary flag)** — A patient who would *not*
   be readmitted receives additional follow-up.  This is far less
   harmful: the extra phone call or clinic visit has minimal downside
   and may even improve patient satisfaction.

Because the asymmetry clearly favours **catching more true positives**
(high recall) over avoiding false alarms, we optimise the threshold to
maximise recall while keeping precision above a clinically acceptable
floor.

## Chosen Operating Point

| Metric    | Value  |
|-----------|--------|
| Threshold | 0.086  |
| Precision | 0.1518 |
| Recall    | 0.7799 |
| F1 Score  | 0.2541 |

## Business Impact

At a threshold of **0.086**, the model catches
**78.0%** of patients who will be readmitted
within 30 days.  The trade-off is that **84.8%**
of flagged patients are false positives — they would not actually have
been readmitted.  For a care-management programme, this is an
acceptable rate: the cost of an unnecessary follow-up call is trivial
compared to the cost of a preventable readmission (~$15,000 average).
