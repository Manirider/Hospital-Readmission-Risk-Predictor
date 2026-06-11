# Data Leakage Audit Report

## Summary

| Metric | Count |
|---|---|
| Dataset rows (before filtering) | 101,766 |
| Expired patients removed | 1,652 |
| Hospice patients removed | 771 |
| **Total rows removed** | **2,423** |
| Dataset rows (after filtering) | 99,343 |

## Leakage Sources

### discharge_disposition_id (values 11, 19, 20, 21)

**Reason:** These codes indicate that the patient **expired** (died) during the encounter.  A deceased patient cannot be readmitted, so keeping these rows would give the model a trivially-predictable negative class and inflate specificity.

**Action:** Remove rows.

### discharge_disposition_id (values 13, 14)

**Reason:** Codes 13 and 14 indicate discharge to **hospice** (home or medical facility).  Hospice patients have elected comfort-only care and are virtually never readmitted.  Including them introduces a near-deterministic signal unrelated to the clinical features we want the model to learn.

**Action:** Remove rows.

### discharge_disposition_id (general)

**Reason:** Even after removing expired / hospice rows, the remaining discharge disposition values (e.g. 'Discharged to home' vs. 'Transferred to another short-term hospital') carry some post-encounter information.  However, this disposition is typically known *at* discharge — the same moment a care team would use the model — so retaining it is defensible for a discharge-time predictor.  We flag this decision explicitly.

**Action:** Retain (with documentation).

## Rationale

In hospital readmission prediction the goal is to estimate, *at the
moment of discharge*, how likely a patient is to return within 30 days.
Any feature that encodes whether the patient *actually* returned — or
whether they were even *able* to return — constitutes information
leakage.  Deceased and hospice patients fall squarely into the latter
category: they cannot be readmitted, so including them gives the model
a cheap shortcut that would not generalise to the intended use case.

We retain the remaining `discharge_disposition_id` values because
discharge destination (e.g. home vs. skilled nursing facility) is known
at the point of discharge and is clinically relevant to readmission
risk.  This is a deliberate, documented design choice.
