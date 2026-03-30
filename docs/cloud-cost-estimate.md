# Cloud Cost Estimate -- Evolve-Trader AI

> Last updated: 2026-03-29
>
> All prices are monthly unless noted. Prices reflect publicly listed rates
> as of March 2026 and may change. EUR amounts converted at ~1 EUR = 1.08 USD.

---

## 1. Resource Requirements by Phase

| Phase Group | Components | vCPU | RAM | Disk | Notes |
|---|---|---|---|---|---|
| **0-1 (Local)** | Docker Compose on dev machine | -- | -- | -- | No cloud cost |
| **2-6 (Core)** | FastAPI + PostgreSQL + React dashboard + signal ingestion | 2-4 | 4-8 GB | 40-80 GB SSD | Single process; DB is the heaviest consumer |
| **7-8 (Orchestrator)** | + Orchestrator + strategy incubator (5-10 LLM agents) | 4-8 | 16-32 GB | 80-160 GB SSD | CPU-bound during evolution cycles; RAM for concurrent LLM context assembly |
| **9+ (Microservices)** | Multiple FastAPI signal-source services | 8+ | 32+ GB | 160+ GB SSD | Horizontal scaling; consider container orchestration |
| **11 (Live)** | All above + live trading engine | 8+ | 32+ GB | 160+ GB SSD | High uptime 9:30-16:00 ET; sub-second latency matters |

---

## 2. Provider Comparison

### Option A: Simple VPS (Hetzner)

Hetzner offers the best price-to-performance ratio for predictable workloads.
All plans include 20 TB outbound traffic.

#### Shared vCPU (CX -- good for Phase 2-6)

| Plan | vCPU | RAM | SSD | Price (EUR/mo) | ~USD/mo |
|---|---|---|---|---|---|
| CX22 | 2 | 4 GB | 40 GB | 3.79 | ~4.10 |
| CX32 | 4 | 8 GB | 80 GB | 6.80 | ~7.35 |

#### Dedicated vCPU (CCX -- good for Phase 7+)

| Plan | vCPU | RAM | SSD | Price (EUR/mo) | ~USD/mo |
|---|---|---|---|---|---|
| CCX13 | 2 | 8 GB | 80 GB | 12.49 | ~13.50 |
| CCX23 | 4 | 16 GB | 160 GB | 24.49 | ~26.45 |
| CCX33 | 8 | 32 GB | 240 GB | 48.49 | ~52.40 |

**Database:** Self-managed PostgreSQL on the same server or a second small VPS.
Hetzner does not offer managed PostgreSQL.

**Phase 2-6 estimate:** CX32 + self-managed Postgres = **~$7/mo**
**Phase 7-8 estimate:** CCX23 or CCX33 = **$26-52/mo**

**Pros:**
- Extremely low cost (5-10x cheaper than AWS/GCP for equivalent specs)
- 20 TB traffic included; no surprise bandwidth bills
- Predictable flat monthly billing
- EU and US (Ashburn) data centers available

**Cons:**
- No managed database -- you handle backups, upgrades, failover
- No built-in auto-scaling or container orchestration
- Smaller ecosystem of managed services
- US availability limited to Ashburn, VA (fine for US market hours)

---

### Option B: AWS (EC2 + RDS)

#### EC2 Instances (us-east-1, On-Demand)

| Instance | vCPU | RAM | Price/hr | ~USD/mo |
|---|---|---|---|---|
| t3.small | 2 | 2 GB | $0.0208 | ~$15 |
| t3.medium | 2 | 4 GB | $0.0416 | ~$30 |
| t3.large | 2 | 8 GB | $0.0832 | ~$60 |
| m6i.large | 2 | 8 GB | $0.096 | ~$70 |
| m6i.xlarge | 4 | 16 GB | $0.192 | ~$140 |

#### RDS PostgreSQL (us-east-1, Single-AZ)

| Instance | vCPU | RAM | Price/hr | ~USD/mo |
|---|---|---|---|---|
| db.t3.micro | 2 | 1 GB | $0.018 | ~$13 |
| db.t3.small | 2 | 2 GB | $0.036 | ~$26 |
| db.t3.medium | 2 | 4 GB | $0.068 | ~$50 |

Storage: ~$0.115/GB/month (gp3). 20 GB = ~$2.30/mo.

#### ECS/Fargate Alternative
Fargate pricing: $0.04048/vCPU/hr + $0.004445/GB/hr.
A 2 vCPU / 4 GB task running 24/7 = ~$100/mo (more expensive than EC2).

**Phase 2-6 estimate:** t3.medium + db.t3.micro + 20GB storage = **~$46/mo**
**Phase 7-8 estimate:** m6i.xlarge + db.t3.medium + 50GB storage = **~$196/mo**

**Pros:**
- Managed RDS with automated backups, failover, point-in-time recovery
- Vast ecosystem: CloudWatch, IAM, Secrets Manager, etc.
- Auto-scaling groups, load balancers available when needed
- Free tier available for first 12 months (t3.micro, 20 GB RDS)

**Cons:**
- 5-10x more expensive than Hetzner for equivalent compute
- Pricing complexity: data transfer, IOPS, EBS, NAT gateway fees add up
- Easy to accidentally overspend without budget alerts
- Overkill for Phase 2-6 workloads

---

### Option C: GCP (Compute Engine + Cloud SQL)

#### Compute Engine (us-central1, On-Demand)

| Instance | vCPU | RAM | ~USD/mo |
|---|---|---|---|
| e2-small | 0.5-2 | 2 GB | ~$12 |
| e2-medium | 1-2 | 4 GB | ~$24 |
| e2-standard-2 | 2 | 8 GB | ~$49 |
| e2-standard-4 | 4 | 16 GB | ~$98 |

Sustained-use discounts apply automatically (~20-30% for always-on workloads).

#### Cloud SQL PostgreSQL

| Tier | vCPU | RAM | ~USD/mo |
|---|---|---|---|
| db-f1-micro | Shared | 0.6 GB | ~$10 |
| db-g1-small | Shared | 1.7 GB | ~$26 |
| db-custom-1-3840 | 1 | 3.75 GB | ~$50 |

Storage: ~$0.17/GB/month (SSD). 20 GB = ~$3.40/mo.

**Phase 2-6 estimate:** e2-medium + db-f1-micro + 20GB = **~$37/mo**
**Phase 7-8 estimate:** e2-standard-4 + db-custom-1-3840 + 50GB = **~$157/mo**

**Pros:**
- Sustained-use discounts applied automatically (no commitment needed)
- Cloud SQL is fully managed with automated backups
- Cloud Run option for microservices (pay-per-request, good for Phase 9+)
- $300 free credits for new accounts

**Cons:**
- Still 4-7x more expensive than Hetzner
- Cloud SQL shared-core instances not covered by SLA
- Fewer data center options than AWS
- Pricing complexity similar to AWS

---

## 3. Provider Cost Summary

| Phase | Hetzner (VPS) | AWS (EC2+RDS) | GCP (CE+SQL) |
|---|---|---|---|
| **2-6 (Core)** | **~$7/mo** | ~$46/mo | ~$37/mo |
| **7-8 (Orchestrator)** | **~$26-52/mo** | ~$196/mo | ~$157/mo |
| **9+ (Microservices)** | ~$52-100/mo | ~$300+/mo | ~$250+/mo |

---

## 4. LLM API Cost Estimate

### Model Pricing (as of March 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best for |
|---|---|---|---|
| Claude Sonnet 4 | $3.00 | $15.00 | Strategy execution, evolution, orchestrator |
| Claude Haiku 3.5 | $0.80 | $4.00 | Signal processing, lightweight classification |
| GPT-4o mini | $0.15 | $0.60 | Bulk signal preprocessing (cheapest option) |

### Token Usage Estimates per Activity

Assumptions: 1 token ~= 4 characters. Context windows include market data, strategy
code, and historical performance summaries.

#### Per-Trade Strategy Execution
- Input: ~4,000 tokens (market context + strategy prompt + recent signals)
- Output: ~1,500 tokens (trade rationale + parameters)
- Model: Claude Sonnet 4
- Cost per trade: (4K x $3.00 + 1.5K x $15.00) / 1M = **$0.035/trade**

#### Evolution Cycle (FIX / DERIVED / CAPTURED)
- Input: ~12,000 tokens (strategy code + performance history + mutation prompt)
- Output: ~4,000 tokens (mutated strategy + reasoning)
- Model: Claude Sonnet 4
- Cycles per week: ~20 (across all strategy types)
- Cost per cycle: (12K x $3.00 + 4K x $15.00) / 1M = **$0.096/cycle**
- Weekly cost: **$1.92/week**

#### Orchestrator Weekly Review
- Input: ~20,000 tokens (portfolio summary + all strategy performances + market regime)
- Output: ~6,000 tokens (allocation changes + reasoning)
- Model: Claude Sonnet 4
- Cost per review: (20K x $3.00 + 6K x $15.00) / 1M = **$0.15/review**
- Weekly cost: **$0.15/week**

#### Incubator Candidate Generation (5-10 concurrent agents)
- Input per agent: ~8,000 tokens (market hypothesis + constraints)
- Output per agent: ~3,000 tokens (candidate strategy)
- Model: Claude Sonnet 4
- Agents per batch: 8 (average), 2 batches/week
- Cost per batch: 8 x (8K x $3.00 + 3K x $15.00) / 1M = **$0.55/batch**
- Weekly cost: **$1.10/week**

#### Signal Processing
- Input: ~1,500 tokens per signal (raw data + classification prompt)
- Output: ~300 tokens per signal (structured signal)
- Model: Claude Haiku 3.5 (cost-optimized) or GPT-4o mini
- Signals per week: ~200

Using Haiku 3.5:
- Weekly cost: 200 x (1.5K x $0.80 + 0.3K x $4.00) / 1M = **$0.48/week**

Using GPT-4o mini:
- Weekly cost: 200 x (1.5K x $0.15 + 0.3K x $0.60) / 1M = **$0.08/week**

### Monthly LLM Cost Summary (10-20 trades/week, moderate activity)

| Activity | Weekly | Monthly |
|---|---|---|
| Strategy execution (15 trades/wk avg) | $0.53 | **$2.10** |
| Evolution cycles | $1.92 | **$7.68** |
| Orchestrator reviews | $0.15 | **$0.60** |
| Incubator generation | $1.10 | **$4.40** |
| Signal processing (Haiku) | $0.48 | **$1.92** |
| **Total** | **$4.18** | **$16.70** |

Using GPT-4o mini for signal processing instead of Haiku reduces the
total to ~$15.10/mo.

### LLM Cost Scaling Notes

- These estimates assume moderate activity. Heavy backtesting or
  large-scale evolution runs could 3-5x the evolution/incubator costs.
- Prompt caching (available for Claude) can reduce input costs by ~90%
  for repeated system prompts, potentially saving $2-4/mo.
- Batch API (50% discount) is viable for non-time-sensitive evolution
  and incubation work, potentially saving another $3-5/mo.
- At scale (Phase 11, live trading), expect $25-50/mo in LLM costs
  with higher trade frequency and more aggressive evolution.

---

## 5. Total Monthly Cost Estimate

| Phase | Infra (Hetzner) | LLM API | Total |
|---|---|---|---|
| **2-6 (Core)** | ~$7 | ~$17 | **~$24/mo** |
| **7-8 (Orchestrator)** | ~$40 | ~$25 | **~$65/mo** |
| **9+ (Microservices)** | ~$75 | ~$35 | **~$110/mo** |
| **11 (Live trading)** | ~$75 | ~$40 | **~$115/mo** |

LLM API costs dominate the budget at Phase 2-6. Infrastructure costs
catch up only at Phase 7+.

---

## 6. Recommendation

### Phase 2-6: Hetzner CX32 (self-managed)

**Recommended setup:**
- **Compute:** Hetzner CX32 -- 4 vCPU, 8 GB RAM, 80 GB SSD (~$7/mo)
- **Database:** PostgreSQL installed directly on the same server
- **Region:** Ashburn, VA (closest to US markets)
- **LLM:** Claude Sonnet 4 for strategy/evolution, Haiku 3.5 for signals

**Why Hetzner over AWS/GCP:**
1. **Cost:** $7/mo vs $37-46/mo -- saves $30-40/mo with no functional difference
   at this scale.
2. **Simplicity:** A single VPS with Docker Compose mirrors the local dev
   environment. No IAM policies, VPCs, security groups, or service quotas to
   configure.
3. **Performance:** 4 shared vCPUs + 8 GB RAM is more than adequate for a single
   FastAPI process, PostgreSQL, and a React dashboard.
4. **20 TB traffic included:** No bandwidth cost surprises.

**What you give up:** Managed database backups (set up `pg_dump` cron instead),
auto-scaling (not needed yet), and the broader AWS/GCP service ecosystem
(not needed yet).

### When to Reassess

| Trigger | Action |
|---|---|
| **Entering Phase 7** (orchestrator + concurrent LLM agents) | Upgrade to CCX23 or CCX33 for dedicated vCPU. Consider a separate Hetzner server for PostgreSQL. |
| **Entering Phase 9** (multiple microservices) | Evaluate whether Hetzner's simplicity still holds or whether container orchestration (AWS ECS, GCP Cloud Run) justifies the cost premium. |
| **Entering Phase 11** (live trading) | Strongly consider AWS or GCP for managed database HA, monitoring, and uptime SLAs. The ~$150/mo premium buys operational reliability that matters when real money is at risk. |
| **LLM costs exceed $75/mo** | Audit token usage. Implement prompt caching, use Batch API for non-urgent work, consider fine-tuned smaller models for repetitive classification tasks. |
| **Hetzner April 2026 price adjustment** | Re-check pricing after April 1, 2026 when new Hetzner rates take effect. Unlikely to change the recommendation materially. |

---

## Sources

- [Hetzner Cloud Pricing](https://www.hetzner.com/cloud)
- [Hetzner CCX Dedicated vCPU](https://www.hetzner.com/cloud/general-purpose)
- [Hetzner Price Adjustment (April 2026)](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/)
- [DigitalOcean Droplet Pricing](https://www.digitalocean.com/pricing/droplets)
- [DigitalOcean Managed Database Pricing](https://www.digitalocean.com/pricing/managed-databases)
- [AWS EC2 On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- [AWS RDS PostgreSQL Pricing](https://aws.amazon.com/rds/postgresql/pricing/)
- [GCP Compute Engine e2-standard-2](https://www.economize.cloud/resources/gcp/pricing/compute-engine/e2-standard-2/)
- [GCP Cloud SQL Pricing](https://cloud.google.com/sql/pricing)
- [Anthropic Claude API Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [OpenAI API Pricing](https://openai.com/api/pricing/)
