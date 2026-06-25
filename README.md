# mala-models-on-thayer
# BP Cascade RI

**Forecasting hypertension medication persistence breaks — and routing 
the right helper before they happen.**

Built for the [RI-AI4H Data Challenge 2026](https://bcbi.brown.edu/news-events/ri-ai4h-2026) 
(Cardiovascular Disease & Hypertension track), this project predicts 
*when* a treated-hypertensive patient's antihypertensive regimen will 
lapse for 30+ days, then uses SHAP-based driver attribution to route 
each at-risk patient to the right intervention — a pharmacist for 
regimen confusion, a social worker for a transportation barrier, or a 
bilingual community health worker for a language or engagement gap.

The output is a capacity-capped, bilingual CHW worklist: a name, a 
predicted break window, and a routed action — designed to be the 
shortest possible distance from a model to a Tuesday-morning phone call.

**Dataset:** [SyntheticRI](https://doi.org/10.26300/g7zj-m980) — a 
Synthea-generated synthetic EHR dataset of 300,000 patients reflecting 
Rhode Island demographics, provided via the Brown Digital Repository.

**Team:** Umar · [Your name] · Chris · Alec · Annie

> Note: built on synthetic data. Results demonstrate methodology and 
> workflow; real-world validation on RI clinical data is the explicit 
> next step.
