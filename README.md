```text
 ███████╗██████╗  ██████╗  ██████╗██╗   ██╗██╗  ██╗██╗███╗   ██╗ ██████╗
 ██╔════╝██╔══██╗██╔═══██╗██╔════╝╚██╗ ██╔╝██║ ██╔╝██║████╗  ██║██╔════╝
 █████╗  ██████╔╝██║   ██║██║  ███╗╚████╔╝ █████╔╝ ██║██╔██╗ ██║██║  ███╗
 ██╔══╝  ██╔══██╗██║   ██║██║   ██║ ╚██╔╝  ██╔═██╗ ██║██║╚██╗██║██║   ██║
 ██║     ██║  ██║╚██████╔╝╚██████╔╝  ██║   ██║  ██╗██║██║ ╚████║╚██████╔╝
 ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
```

# ai/ml engineer · building production ai systems

![profile views](https://komarev.com/ghpvc/?username=FrogyKing&color=00d9ff&style=flat-square&label=profile+views)
![focus](https://img.shields.io/badge/focus-deep_ai_%7C_ml_systems-00d9ff?style=flat-square)
![status](https://img.shields.io/badge/currently-master%27s_in_ai-blueviolet?style=flat-square)

### deep ai · applied math · ml systems

hey, i'm frogyking. i like getting close to the math, close to the hardware, and close to what happens after deployment.

my lane: understanding how models learn, how inference moves through a gpu, and how an ml system earns its place in production. clean experiments, explicit tradeoffs, and evidence before hype. that's the vibe.

## `> whoami`

currently pursuing a **master's degree in artificial intelligence**. connecting the theory to the engineering, one experiment at a time.

```text
the math       → optimization, probability, linear algebra, numerical stability
the model      → representations, objectives, generalization, evaluation
the machine    → memory traffic, parallelism, latency, throughput
the system     → reproducibility, observability, cost, recovery
```

## 🚀 building right now

### ⚡ 01 / [helix-gpu-kernels](https://github.com/FrogyKing/helix-gpu-kernels)

**gpu inference optimization with cuda c++ and triton · active development**

this is where i go below the model api: into kernels, memory access patterns, and the constraints that decide whether an optimization actually helps.

the project centers on cuda c++ / triton inference work and benchmarks on modal. the engineering questions i'm working through:

- **where is the bottleneck?** reason about arithmetic intensity, memory bandwidth, and compute limits before choosing an optimization.
- **what belongs together?** explore tiling, data reuse, and fusion while accounting for register pressure and occupancy.
- **does it stay correct?** compare against a reference with dtype-aware numerical tolerances; speed only matters when the output still holds up.
- **does the comparison hold?** make hardware, shapes, dtypes, warmup, synchronization, and timing methodology explicit.

```text
benchmark discipline

reference → correctness check → warmup → synchronized timing → comparison

record: gpu / software versions / shape / dtype / workload
inspect: latency distribution / throughput / numerical error
speedup = reference latency / candidate latency
          for the same workload and measurement conditions
```

the goal is faster inference with a defensible explanation of why. performance gains stay unclaimed until reproducible measurements support them.

### 🛡️ 02 / [sentinel-mlops-aws](https://github.com/FrogyKing/sentinel-mlops-aws)

**cost-conscious production ml lifecycle on aws · active development**

a model release is a decision under uncertainty. sentinel is where i'm working on the lifecycle around that decision: how to evaluate a candidate, observe changing data, limit rollout risk, and recover without making the cloud bill the main character.

the design priorities:

- **traceability:** connect data versions, training configuration, evaluation results, and model artifacts.
- **drift:** distinguish changes in input distributions from changes in predictive quality; a drift alert is a signal to investigate, not proof that retraining will help.
- **champion–challenger:** compare candidates against the incumbent using consistent evaluation criteria and relevant data slices.
- **canary rollout and rollback:** expand traffic gradually, observe quality and operational signals, and define when to return to the previous version.
- **cost discipline:** treat compute, storage, monitoring, and idle capacity as part of the design problem.

```text
intended lifecycle

versioned data → train → evaluate challenger → promotion gate
                                                  ↓
monitor ← staged rollout ← canary ← candidate release
   │                          │
   ├─ drift → investigate     └─ guardrail breach → rollback
   └─ feedback → next evaluation cycle
```

the target is a lifecycle with clear promotion criteria, observable behavior, and a credible recovery path. these are active development priorities, not a claim that every component is already complete.

## 🧩 more projects

### 👁️ 03 / [deepseek-ocr-app](https://github.com/FrogyKing/deepseek-ocr-app)

a document intelligence application powered by deepseek-ocr-2: structured markdown from scanned documents, plus annotated images with detected regions and bounding boxes. remote gpu inference on modal connects to an asynchronous fastapi backend and a streamlit interface.

`document ai` · `computer vision` · `modal gpu` · `fastapi` · `streamlit`

### 🐟 04 / anchoveta-mlops

an end-to-end mlops framework on google cloud for fish-density prediction. domain assumptions, reproducible pipelines, drift detection, and walk-forward validation meet in one applied problem.

`vertex ai pipelines` · `bigquery` · `docker` · `ks-test` · `walk-forward validation`

### 📞 05 / telco-complaints-rag-classifier

a project around retrieval-augmented classification of telco complaints. the technical questions i care about: retrieval relevance, ambiguous labels, evidence grounding, and separating retrieval errors from classification errors.

`rag` · `gcp` · `langfuse` · `nlp`

## 🛠️ tech stack

<details>
<summary>open the toolbox — languages, inference, agents, cloud, and observability</summary>

### ⚡ languages & gpu

![python](https://img.shields.io/badge/python-3776ab?style=flat-square&logo=python&logoColor=white)
![c++](https://img.shields.io/badge/c%2B%2B-00599c?style=flat-square&logo=cplusplus&logoColor=white)
![cuda](https://img.shields.io/badge/cuda-76b900?style=flat-square&logo=nvidia&logoColor=white)
![triton](https://img.shields.io/badge/triton-222222?style=flat-square)

### 🤖 generative ai & agents

![langchain](https://img.shields.io/badge/langchain-1c3c3c?style=flat-square&logo=langchain&logoColor=white)
![langgraph](https://img.shields.io/badge/langgraph-1c3c3c?style=flat-square&logo=langgraph&logoColor=white)
![google adk](https://img.shields.io/badge/google_adk-4285f4?style=flat-square&logo=google&logoColor=white)

### 👁️ serving & applications

![fastapi](https://img.shields.io/badge/fastapi-009688?style=flat-square&logo=fastapi&logoColor=white)
![modal](https://img.shields.io/badge/modal-000000?style=flat-square&logo=modal&logoColor=white)
![streamlit](https://img.shields.io/badge/streamlit-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![hugging face](https://img.shields.io/badge/hugging_face-ffd21e?style=flat-square&logo=huggingface&logoColor=black)

### 🔁 mlops & data

![vertex ai](https://img.shields.io/badge/vertex_ai-4285f4?style=flat-square&logo=googlecloud&logoColor=white)
![bigquery](https://img.shields.io/badge/bigquery-669df6?style=flat-square&logo=googlebigquery&logoColor=white)
![docker](https://img.shields.io/badge/docker-2496ed?style=flat-square&logo=docker&logoColor=white)
![terraform](https://img.shields.io/badge/terraform-7b42bc?style=flat-square&logo=terraform&logoColor=white)

### ☁️ cloud & observability

![google cloud](https://img.shields.io/badge/google_cloud-4285f4?style=flat-square&logo=googlecloud&logoColor=white)
![aws](https://img.shields.io/badge/aws-232f3e?style=flat-square)
![langfuse](https://img.shields.io/badge/langfuse-000000?style=flat-square&logo=langfuse&logoColor=white)

</details>

## 🧠 how i think about the work

**start with the objective.** the loss function, the evaluation metric, and the operational goal need to make sense together. optimizing a proxy is easy; checking whether it represents the actual problem takes more care.

**respect the numerics.** precision, conditioning, accumulation order, and tolerances are part of the algorithm once it meets hardware.

**evaluate beyond an average.** leakage, imbalance, calibration, distribution shift, and weak data slices can hide behind a clean aggregate score.

**make systems inspectable.** a useful experiment should explain what changed. a useful deployment should explain what is running, how it is behaving, and how to recover.

```text
my engineering loop

form a hypothesis
    → establish a baseline
    → change one thing deliberately
    → measure correctness, quality, latency, and cost as relevant
    → inspect failure cases
    → keep what the evidence supports
```

good math. honest benchmarks. systems you can reason about. let's build.

```text
[ status ]  building → measuring → shipping → monitoring → repeat
```

📫 into gpu kernels, document ai, or mlops? let's talk shop.
