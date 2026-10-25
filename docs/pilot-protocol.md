# Housing energy decision-support pilot protocol — proposed, not executed

## Decision and participant

Proposed use: help a housing/energy analyst explore EPC data quality, authority
trends and model uncertainty when planning further investigation. This is not an
automated retrofit eligibility, regulatory compliance or funding decision.

The user must identify an actual participating organisation and an appropriate
analyst, agree the decision being supported, and authorize any outreach. No
organisation, participant, user count or adoption is currently claimed.

## Reviewable pilot package

Provide a versioned dataset manifest, model card, known limitations, reproducible
example queries, read-only dashboard/API and task worksheet. Show certificate
coverage and missingness alongside predictions; never present estimated values
as observed improvements or real energy-bill savings.

Suggested analyst tasks, to be agreed with the participant:

1. Compare authority EPC distributions and identify differences in coverage.
2. Explain a below-C prediction using the actual selected model's explanation.
3. Identify a subgroup where uncertainty or missingness makes a conclusion weak.
4. Produce a short findings note with the correct dataset period and limitations.

## Measurement plan to freeze before sessions

- Baseline: participant's current workflow completing the same agreed tasks.
- Intervention: project workflow with the same input data and task requirements.
- Use matched task variants and counterbalance order to reduce practice effects.
- Record task completion, errors against an agreed answer rubric, time per task,
  requests for clarification, and participant feedback. Retain unsuccessful tasks.
- Report participant and task counts, individual results, exclusions, protocol
  deviations and descriptive uncertainty. A small convenience sample does not
  establish population-wide productivity gains.
- Separate offline prediction quality from analyst usability and downstream
  business outcomes. Real retrofit savings require subsequent interventions and
  a separate measurement design.

## Evidence records

Retain authorized session dates, anonymised participant roles, task/rubric version,
dataset/model/run IDs, baseline and intervention measurements, feedback and
analysis code. Obtain participant consent for any quotes or identifiable details.
Keep records out of a public repository. Do not invent participants to fill gaps.

## Completion gates

1. Organisation and participant identified; scope and access agreed.
2. Package ready, instructions checked, protocol frozen.
3. Sessions actually completed; raw measurements reviewed.
4. Results analysed with limitations; participant review where appropriate.
5. Resume claims restricted to the measured pilot outcomes and sample size.

Until these gates are met, safe wording is "prepared a stakeholder pilot protocol",
not "used by housing authorities" or "reduced analyst time".
