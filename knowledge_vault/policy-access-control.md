---
id: policy-access-control
title: Access control policy
tenant: tenant_a
domain: policy
risk_tier: TIER_1
version: 3
links: [policy-assistant-scope, sop-hold-reject, glossary]
---
There are three user roles: operator, reviewer and tenant admin. Operators can ask questions, look up lots and file requests. Reviewers approve or reject requests. Tenant admins manage users and roles for their own tenant.

A reviewer can never approve a request that they filed themselves. This is the four-eyes rule. Requests that nobody has decided within 24 hours expire and are treated as rejected.

Users only ever see data from their own tenant. The assistant limits what it does for a user in line with [[policy-assistant-scope]].
