---
id: sop-sampling-testing
title: Sampling and testing procedure
tenant: tenant_a
domain: quality
risk_tier: TIER_2
version: 3
links: [sop-hold-reject, sop-lot-release]
---
For every lot of up to 1000 units, 5 units are sampled at random. For lots of more than 1000 units, 10 units are sampled. Samples are tested under protocol PROT-TST-2.

Test results must be recorded within 5 business days of sampling. While results are pending the lot is placed ON_HOLD under [[sop-hold-reject]].

A single failed sample puts the lot on hold and doubles the sample size for a retest. A lot with two failed retests is rejected. Passing results are one of the inputs to [[sop-lot-release]].
