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

i'm **luis gonzalez**, an ai/ml engineer with **seven years of experience** building solutions across industries. currently pursuing a **master's degree in artificial intelligence** — connecting the theory to the engineering, one experiment at a time.

[portfolio ↗](https://frogyking.github.io/) · [linkedin ↗](https://www.linkedin.com/in/luis-gonzalez-alvarez/)

```text
the math       → optimization, probability, linear algebra, numerical stability
the model      → representations, objectives, generalization, evaluation
the machine    → memory traffic, parallelism, latency, throughput
the system     → reproducibility, observability, cost, recovery
```

## 🚀 building right now

### 🔬 01 / [anomavision](https://github.com/FrogyKing/anomavision)

**industrial anomaly detection · dense visual representations · defect localization**

teach a system what normal looks like, then inspect what breaks the pattern. anomavision connects frozen dinov3 patch features, compact reference memory banks, and nearest-neighbor scoring to an industrial visual inspection workflow.

the engineering behind the inspection:

- **learn from normal examples:** build a patchcore-style memory bank from defect-free training images, compressed with greedy k-center coresets.
- **localize the exception:** turn patch distances into defect heatmaps and overlays; optional sam 3 foreground gating helps suppress background responses.
- **keep the evidence close:** retrieve similar normal references with faiss, calibrate thresholds on held-out normal images, and monitor embedding drift.
- **connect the pieces:** fastapi serving, a gradio inspection console, modal gpu execution, and reproducible experiment artifacts.

```text
normal images → dinov3 patch features → coreset memory bank
                                             ↓
new image → patch distances → anomaly heatmap → inspection decision
                                  + similar normal references
                                  + embedding drift monitoring
```

**published results:** image auroc **1.0000** on mvtec ad `bottle` and **0.9995** on `metal_nut`. these results cover two benchmark categories, with unsupervised threshold calibration. the report also examines where foreground gating helps and where it clips defect boundaries.

`computer vision` · `dinov3` · `patchcore` · `faiss` · `sam 3` · `modal` · `fastapi`

[inspection outputs & methodology ↗](https://github.com/FrogyKing/anomavision#7-sample-real-inspection-outputs--qualitative-failure-analysis) · [evaluation results ↗](https://github.com/FrogyKing/anomavision/blob/main/artifacts/evaluation_summary.json)

### 🤖 02 / [pushlab](https://github.com/FrogyKing/pushlab)

**robust reinforcement learning · quadruped locomotion · gpu physics**

what does a robot learn when the world pushes back? pushlab trains a quadruped to recover from external disturbances, combining ppo, gpu-parallel mujoco mjx environments, and an interactive 3d robotics lab.

the comparison is controlled: nominal and disturbance-trained policies share the architecture, observations, reward, hyperparameters, and training horizon.

- **train in parallel:** 512 mjx environments on a modal nvidia l4, with jax and optax.
- **test generalization:** evaluate push intensity, unseen directions, and out-of-distribution disturbances.
- **look beyond survival:** inspect return, recovery time, velocity tracking, and the physical limits of the controller.
- **make it explorable:** connect a fastapi simulation backend to a react / three.js laboratory.

```text
observe → policy → joint actions → simulated physics → reward
   ↑                                  │
   └────────── next observation ───────┘

nominal ppo vs. robust ppo
same training horizon → shared evaluation → inspect recovery
```

**published results:** under **52–64 n** pushes, robust ppo achieved **100% survival**, compared with **41.7%** for baseline ppo, over 12 simulation seeds per regime. extreme out-of-distribution pushes reduced survival to 66.7% vs. 25.0%. these are simulation results.

`reinforcement learning` · `ppo` · `jax` · `mujoco mjx` · `modal` · `three.js`

[benchmarks & robustness curves ↗](https://github.com/FrogyKing/pushlab#1-research-question--core-findings) · [mathematical formulation ↗](https://github.com/FrogyKing/pushlab/blob/main/docs/MATHEMATICS.md)

### 🔎 03 / [omniseek-multimodal](https://github.com/FrogyKing/omniseek-multimodal)

**one search experience across text, images, audio, video, and code**

find the signal across a collection of files. omniseek uses embeddinggemma 2 and on-demand modal gpus to bring five modalities into a shared embedding space, with a local fastapi backend and a react interface.

the interesting part is what stays attached to the result:

- **source-aware retrieval:** preserve original files, code locations, and media timestamps.
- **native multimodal inputs:** search with text or images and preview matches across supported modalities.
- **inspectable representations:** explore index statistics and similarity behavior in the embedding lab.
- **bounded gpu use:** keep existing indexes searchable on cpu after cloud workers stop; cached demo queries can run locally.

```text
text / image / audio / video / code
                ↓
       multimodal embeddings
                ↓
         shared vector index
                ↓
query → similarity search → source + preview + location
```

**published evidence:** real indexed demo assets and a recorded live api validation covering indexing, a new gpu-backed query, and a restart without gpu access. arbitrary new query embeddings require model inference.

`multimodal ai` · `embeddinggemma 2` · `faiss` · `modal` · `fastapi` · `react`

[application & screenshots ↗](https://github.com/FrogyKing/omniseek-multimodal) · [api validation ↗](https://github.com/FrogyKing/omniseek-multimodal/blob/main/experiments/results/live-api-validation.json)

### ⚡ 04 / [helix-gpu-kernels](https://github.com/FrogyKing/helix-gpu-kernels)

**gpu inference optimization with cuda c++ and triton · measured l4 experiments**

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

**published evidence:** the initial nvidia l4 run passed **176 correctness tests** and validated **171 benchmark comparisons**. the report includes faster softmax / rmsnorm cases and slower gemm / attention kernels, with explicit shapes, precision, and timing conditions.

the goal is faster inference with a defensible explanation of why. kernel measurements and full-model generation remain separate comparisons.

`cuda c++` · `triton` · `pytorch` · `numerical analysis` · `modal`

[benchmark report ↗](https://github.com/FrogyKing/helix-gpu-kernels/blob/main/benchmarks/reports/initial-l4.md)

### 🛡️ 05 / [sentinel-mlops-aws](https://github.com/FrogyKing/sentinel-mlops-aws)

**cost-conscious ml lifecycle · locally verified · aws deployment defined**

a model release is a decision under uncertainty. sentinel connects reproducible fraud-model training, chronological evaluation, versioned artifacts, drift monitoring, and real http inference. the engineering question: how do you evaluate a candidate, observe changing data, control promotion, and recover without making the cloud bill the main character?

the design priorities:

- **traceability:** connect data versions, training configuration, evaluation results, and model artifacts.
- **drift:** distinguish changes in input distributions from changes in predictive quality; a drift alert is a signal to investigate, not proof that retraining will help.
- **champion–challenger:** compare candidates against the incumbent using consistent evaluation criteria and relevant data slices.
- **release and recovery:** exercise local promotion and rollback; define aws canary routing and operational checks in terraform and the runbook.
- **cost discipline:** treat compute, storage, monitoring, and idle capacity as part of the design problem.

```text
target aws lifecycle

versioned data → train → evaluate challenger → promotion gate
                                                  ↓
monitor ← staged rollout ← canary ← candidate release
   │                          │
   ├─ drift → investigate     └─ guardrail breach → rollback
   └─ feedback → next evaluation cycle
```

**published evidence:** **0.7566 pr-auc** on the chronological held-out public fraud benchmark, plus locally verified inference and recovery behavior. the baseline achieved lower held-out test cost — a useful reminder to inspect the operational metric alongside model quality.

the local lifecycle is implemented and verified. aws infrastructure is defined and terraform-validated; cloud deployment and canary traffic remain unverified.

`mlops` · `scikit-learn` · `fastapi` · `dvc` · `mlflow` · `aws` · `terraform`

[verification record ↗](https://github.com/FrogyKing/sentinel-mlops-aws/blob/main/docs/verification.md)

## 🧩 more projects

### 👁️ 06 / [deepseek-ocr-app](https://github.com/FrogyKing/deepseek-ocr-app)

a document intelligence application powered by deepseek-ocr-2: structured markdown from scanned documents, plus annotated images with detected regions and bounding boxes. remote gpu inference on modal connects to an asynchronous fastapi backend and a streamlit interface.

`document ai` · `computer vision` · `modal gpu` · `fastapi` · `streamlit`

### 🐟 07 / [anchoveta-mlops](https://github.com/FrogyKing/anchoveta-mlops)

an end-to-end mlops framework on google cloud for fish-density prediction. domain assumptions, reproducible pipelines, drift detection, and walk-forward validation meet in one applied problem.

`vertex ai pipelines` · `bigquery` · `docker` · `ks-test` · `walk-forward validation`

### 📞 08 / [telco-complaints-rag-classifier](https://github.com/FrogyKing/telco-complaints-rag-classifier)

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

📫 into computer vision, reinforcement learning, multimodal ai, gpu kernels, or mlops? [let's talk shop ↗](https://www.linkedin.com/in/luis-gonzalez-alvarez/)
