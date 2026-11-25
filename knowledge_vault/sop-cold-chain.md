---
id: sop-cold-chain
title: Cold chain storage procedure
tenant: tenant_a
domain: quality
risk_tier: TIER_2
version: 4
links: [sop-temperature-excursion, safety-protocol-lot-verification, catalog-sku-099]
---
Refrigerated products such as SKU-099 must be stored between 2 and 8 degrees Celsius at all times. Storage temperature is logged every 15 minutes by the cold room monitor under protocol PROT-SAF-7.

The cold room has two independent probes. A reading is valid only when both probes agree within 1 degree Celsius. If they disagree, the cold room is treated as out of range until maintenance confirms a probe fault. Out of range readings are handled under [[sop-temperature-excursion]].

Temperature logs are reviewed by the quality team every Monday and are checked again during lot verification, see [[safety-protocol-lot-verification]].
