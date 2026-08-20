# Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference

📄 [**Paper**]([https://arxiv.org/abs/XXXX.XXXXX](https://arxiv.org/abs/2608.18982))

![Monroe overview: (A) multi-task pretraining on PM6 and PCBA with structural and decorrelation losses under uncertainty weighting; (B) graph construction with featurization and stereochemistry-aware rewiring; (C) in-context probabilistic inference, where attention-pooled embeddings and training labels are passed to TabPFN for a single-pass posterior-predictive distribution.](figure1.png)

Monroe uses GRIT (Graph Inductive Biases in Transformers without Message Passing) for multi-task learning on molecular property prediction.

It is pre-trained jointly on two complementary sources: **PM6** (~81M molecules) supplies dense semi-empirical quantum-chemical properties, and **PCBA** (1.56M molecules) supplies sparse but biologically aligned binary bioassay labels, much closer in distribution to the downstream tasks. Together with a conformer denoising objective, that makes 1,152 pre-training tasks: 62 PM6 graph-level targets, 1,089 PCBA per-assay classifications, and 1 node-level conformation denoising task.

The encoder is evaluated on downstream ADMET (Polaris) and activity-cliff (MoleculeACE) benchmarks.

## Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

To include dependencies for **data preprocessing** (PM6 pipeline):

```bash
uv pip install -e ".[preprocessing]"
```

To include dependencies for **downstream evaluation** (TabPFN, Polaris, MoleculeACE):

```bash
uv pip install -e ".[eval]"
```

Or install everything:

```bash
uv pip install -e ".[preprocessing,eval]"
```

### Environment

The SLURM scripts in `scripts/` source a `.env` file for the virtualenv and a few
environment variables. Copy the template and edit it for your machine:

```bash
cp .env.example .env
```

## Quick Start

The pretrained model ships with the repository (via Git LFS — see
[Pretrained model](#pretrained-model)). Featurize a molecule and embed it —
no arguments needed:

```bash
python monroe/run_inference.py
python monroe/run_inference.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
```
This prints a 720-d graph-level embedding. `load_ckpt` restores the **encoder only**,
so a forward pass gives you a fingerprint rather than task predictions. 

## Evaluation

Monroe is evaluated by freezing the encoder and embedding each molecule. The default
downstream predictor is TabPFN, which predicts *in context*: it conditions on the
labelled training embeddings and predicts test molecules in a single forward pass.

### Step 1: Precompute benchmark graphs (once)

For speed, both benchmarks read cached molecular graphs, so build them before evaluating:

```bash
python -m monroe.preprocessing.build_eval_graphs --benchmark polaris --output-path data/polaris/
python -m monroe.preprocessing.build_eval_graphs --benchmark moleculeace --output-path data/moleculeace/
```

On SLURM this parallelises as an array job:

```bash
sbatch scripts/preprocessing/build_eval_graphs.sbatch --benchmark polaris --output-path data/polaris/
```

Conformer generation fails for a small fraction of molecules, which fall back to a flat
2D layout. `scripts/preprocessing/fix_2d_molecules.py` retries those with relaxed
parameters — worth running, since Monroe reads 3D structure:

```bash
python scripts/preprocessing/fix_2d_molecules.py --benchmark polaris
```

### Step 2: Evaluate a checkpoint

This needs only the graphs from Step 1 — no PM6 data:

```bash
python -m monroe.run_eval --ckpt-path /path/to/checkpoint
python -m monroe.run_eval --ckpt-path ... --use-ema           # evaluate EMA weights
python -m monroe.run_eval --ckpt-path ... --skip-moleculeace  # Polaris only
```

## In-context inference with TabPFN

In Monroe, the downstream predictions come from [TabPFN](https://www.nature.com/articles/s41586-024-08328-6),
a transformer pretrained on synthetic tabular problems that predicts *in context*.
The frozen encoder generates embeddings for both the training and test molecules, and TabPFN
uses them plus labels as context to predict the labels of the test molecules in a single
forward pass. There are no weight updates and no per-task hyperparameters, so adapting to a
new assay costs one forward pass rather than a training run.

### TabPFN weights

The paper reports TabPFN v3, whose weights are licence-gated:

1. Log in (or register) at [ux.priorlabs.ai](https://ux.priorlabs.ai) and accept the licence
   on the **Licenses** tab.
2. Copy the API key from [ux.priorlabs.ai/account](https://ux.priorlabs.ai/account).
3. Put it in your `.env` (see `.env.example`):

```bash
export TABPFN_TOKEN="<your-api-key>"
```

Pick the weights with `TABPFN_MODEL_VERSION=v3` (default) or `=v2`.

## Inference on a new dataset

[`examples/openadmet.ipynb`](examples/openadmet.ipynb) shows how to run Monroe on a new dataset.
The example uses real drug-discovery measurements from **ExpansionRx**,
released through the [OpenADMET](https://openadmet.org/) project as a blind challenge: 9 ADMET
endpoints over ~7,600 molecules, with the organisers' own train/test split.

The two calls it is built around work on any table with a `SMILES` column:

```python
from monroe.eval.embed import embed_smiles
from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn

embeddings = embed_smiles(smiles, encoder)
predictions = fit_predict_tabpfn(X_train, y_train, X_test, is_classification=False,
                                 ensemble_specs=default_ensemble_specs())
```

## Training

Train with `torchrun`; hyperparameters default from `monroe/config.py` (`run_train.py --help`).

```bash
torchrun --nproc_per_node 4 monroe/run_train.py \
  --pm6-dir data/pm6 \
  --pcba-dir data/pcba/train/consolidated
```

Pass both to reproduce our checkpoint; drop `--pcba-dir` for PM6 alone (see
[PCBA Preprocessing](#pcba-preprocessing)).

`--train-node-budget` is the batch size in graph nodes, summed across GPUs (default 100,000).
On a single GPU, lower it to fit memory — e.g. `--train-node-budget 25000` on an 80 GB card.

For SLURM, use `scripts/training/train.sbatch`.

## Data Preprocessing

Monroe is pretrained on the [PubChemQC PM6](https://doi.org/10.1021/acs.jcim.0c00740) dataset (~83M molecules with PM6-level quantum-chemical properties). The raw data comes from two sources:

- **Descriptors (graph-level labels)**: [Graphium](https://graphium-docs.datamol.io/stable/datasets.html) — 20 parquet files of computed molecular properties
- **Conformer geometries (3D coordinates)**: [HuggingFace](https://huggingface.co/datasets/molssiai-hub/pubchemqc-pm6) — PM6-optimized S0 ground-state structures

The preprocessing pipeline converts these into a CSR-sharded format suitable for training.

### Step 1: Download PM6 descriptors

Download the 20 parquet files of molecular descriptors from Zenodo:

```bash
bash scripts/preprocessing/get_pm6_descriptors.sh
# Output: data/pm6/pm6_processed_*.parquet, data/pm6/pm6_random_splits.pt
```

### Step 2: Download PM6 conformer geometries

Download the PM6-optimized conformer geometries from HuggingFace and export to parquet. This fetches the `molssiai-hub/pubchemqc-pm6` dataset (pm6opt config), filters for S0 ground state conformers, and shards into 500 parquet files:

```bash
python scripts/preprocessing/get_pm6_conformers.py
# Output: data/pm6_hf/pm6opt_s0_parquet/*.parquet
```

### Step 3: Prepare and reshard

Deduplicate molecules, filter out overlap with downstream evaluation benchmarks, enrich with geometry from Step 2, and reshard into the target number of parquet files:

```bash
python -m monroe.preprocessing.pm6_prep \
  --out-dir data/pm6_resharded \
  --pm6-pattern "data/pm6/pm6_processed_*.parquet" \
  --geom-dir data/pm6_hf/pm6opt_s0_parquet \
  --num-target-shards 1010
# Output: data/pm6_resharded/shards/*.parquet
```

### Step 4: Build CSR shards

Convert the parquet shards into NumPy CSR arrays (node features, edge indices, coordinates, targets, etc.). This featurizes each molecule using RDKit via `build_single_graph`:

```bash
python -m monroe.preprocessing.build_shards \
  --shards-dir data/pm6_resharded/shards \
  --out-dir data/pm6_monroe/ \
  --use-node-features
# Output: data/pm6_monroe/*.{NF,NC,EI,EC,POS,POS_RDKIT,Y_graph,node_ptr,edge_ptr}.npy
```

On a SLURM cluster, this can be run as a job array for parallelism:

```bash
sbatch scripts/preprocessing/build_shards.sbatch
```

### Step 5: Precompute RRWP

Compute Random Walk Positional Encoding (RRWP) features — node and edge random walk probabilities used as positional encodings in the GRIT encoder:

```bash
python scripts/preprocessing/precompute_rrwp.py \
  --shard-dir data/pm6_monroe/ \
  --walk-len 16 \
  --max-nodes 200
# Output: data/pm6_monroe/*.{16.rrwp_nodes,16.rrwp_edges,log_deg}.npy
```

On SLURM, use the provided array job script:

```bash
sbatch scripts/preprocessing/precompute_rrwp.sbatch
```

### Step 6 (optional): Consolidate shards

Merge many small shards into fewer large ones to reduce filesystem overhead:

```bash
python scripts/preprocessing/consolidate_shards.py \
  --consolidation-factor 10 \
  --input-dir data/pm6_monroe/ \
  --output-dir data/pm6/ \
  --include-rrwp
# Output: data/pm6/*.npy (fewer, larger shards)
```

### Training data format

Each shard prefix has the following files:

| File | Contents |
|------|----------|
| `.NF.npy` | Node float features |
| `.NC.npy` | Node categorical codes |
| `.EI.npy` | Edge index (COO format) |
| `.EC.npy` | Edge categorical codes |
| `.POS.npy` | PM6-optimized 3D coordinates |
| `.POS_RDKIT.npy` | RDKit-generated 3D coordinates |
| `.Y_graph.npy` | Graph-level targets |
| `.node_ptr.npy` | CSR row pointers (nodes per molecule) |
| `.edge_ptr.npy` | CSR row pointers (edges per molecule) |
| `.{walk_len}.rrwp_nodes.npy` | Node RRWP positional encoding |
| `.{walk_len}.rrwp_edges.npy` | Edge RRWP positional encoding |
| `.log_deg.npy` | Log degree for normalization |

## PCBA Preprocessing

Monroe is pre-trained on PM6 **and** PCBA jointly: 1.56M molecules across 1,328 binary
bioassays. Run this to reproduce our checkpoint; skip it for a PM6-only model.

PCBA has no PM6 geometries — conformers are generated with ETKDG+MMFF94 at build
time, and `POS` is left as NaN (positions come from `POS_RDKIT`, forcing `q=0`).

### Step 1: Download PCBA

Get `pcba_1328.zip` from [Zenodo](https://zenodo.org/records/8024997) and unzip it;
the pipeline reads `PCBA_1328_1564k.parquet`.

```bash
# Output: data/pcba_raw/pcba_1328/PCBA_1328_1564k.parquet
```

### Step 2: Prepare and shard

Converts SMILES to InChI, drops molecules overlapping the evaluation benchmarks,
filters assays, and writes train/val parquet shards:

```bash
python -m monroe.preprocessing.pcba_prep \
  --pcba-parquet data/pcba_raw/pcba_1328/PCBA_1328_1564k.parquet \
  --out-dir data/pcba \
  --num-train-shards 20 \
  --val-frac 0.10 \
  --seed 1
# Output: data/pcba/{train,val}/shards/*.parquet,
#         data/pcba/{manifest.json,pcba_stats.json}
```

Assays are dropped unless they have at least 100 positives, 100 negatives, and 1,000
non-NaN labels (`--min-pos-per-assay`, `--min-neg-per-assay`, `--min-valid-per-assay`).
**The defaults reproduce the published dataset: 1,089 of the 1,328 assays survive.**
The exact thresholds used are recorded in `manifest.json` under `filter_thresholds`.

### Step 3: Build CSR shards

Reuses the PM6 builder in `--smiles-only` mode (generates conformers, leaves `POS` NaN):

```bash
python -m monroe.preprocessing.build_shards \
  --shards-dir data/pcba/train/shards \
  --out-dir data/pcba/train/built \
  --smiles-only
python -m monroe.preprocessing.build_shards \
  --shards-dir data/pcba/val/shards \
  --out-dir data/pcba/val/built \
  --smiles-only
```

This is the slow stage. A single process claims and builds every shard in turn.
Shards are claimed through a lock directory, so the step parallelises without
coordination: run N copies of the same command concurrently (as a SLURM array,
say), each with `--exit-after-one`, and each will claim the next unbuilt shard
and exit.

### Step 4: Precompute RRWP

```bash
python scripts/preprocessing/precompute_rrwp.py \
  --shard-dir data/pcba/train/built \
  --walk-len 16 \
  --max-nodes 500
```

### Step 5: Consolidate into one resident shard

PCBA is held resident in shared memory, so it must be a single shard:

```bash
N=$(ls data/pcba/train/built/*.NF.npy | wc -l)
python scripts/preprocessing/consolidate_shards.py \
  --consolidation-factor "$N" \
  --input-dir data/pcba/train/built \
  --output-dir data/pcba/train/consolidated \
  --include-rrwp
# Output: data/pcba/train/consolidated/
```

### Step 6: Train with PCBA

Point training at the consolidated shard:

```bash
python monroe/run_train.py \
  --pm6-dir data/pm6 \
  --pcba-dir data/pcba/train/consolidated
```

See `monroe/config.py --help` for the full list of PCBA options.

## Sharding and data loading

PM6 is split into ~1,000 shards during preprocessing and then consolidated back into ~100
before training. The two steps have opposite goals:

- **Splitting** (`pm6_prep --num-target-shards 1010`) hash-distributes molecules across
  many small shards so the expensive CSR build (RDKit featurization + RRWP) can run as an
  embarrassingly parallel job array — one process per shard.
- **Consolidating** (`consolidate_shards --consolidation-factor 10` → ~101 shards) merges
  them back into fewer, larger files for training. The loader swaps one shard file per
  rotation, so fewer files means less filesystem and per-load overhead.

During training the two sources are handled differently:

- **PM6 rotates.** The loader keeps one shard resident in shared memory and prefetches the
  next one asynchronously; only local rank 0 reads from disk, and the other DDP ranks map
  the same shared memory. Training runs `--n-shards` passes over a shuffled shard order,
  and within each shard batches are formed by a node budget (`--train-node-budget`), not a
  fixed molecule count.
- **PCBA stays resident.** It is loaded once into shared memory at startup and kept for the
  whole run (never rotated out), and a small PCBA mini-batch (`--pcba-mols-per-batch`) is
  sampled and concatenated onto every PM6 batch. So PM6 is the large rotating background and
  PCBA is a persistent, biologically-aligned signal mixed into every step.

## Reproducing results

Generate the publication figures and tables from the results in `results/`:

```bash
python scripts/analysis/generate_figures.py
python scripts/analysis/generate_tables.py --leaderboard
```

These use Tukey HSD pairwise tests with Benjamini-Hochberg FDR correction, and MAE scaling for the Polaris regression tasks.

Figures are saved to `results/figures/` and tables to `results/tables/`. The headline
comparison figure reproduces as:

![Win rates (bars) and per-task pairwise win matrices for the three models introduced here (Monroe, MiniMol_PFN, CheMeleon_PFN) against the leading prior models and a classical gradient-boosting baseline (GBM on ECFP4 + RDKit descriptors), on Polaris (28 tasks) and MoleculeACE (30 tasks). A cell counts the tasks the row model won against the column model.](figure3.png)

## Pretrained model

The released encoder ships in `checkpoint/` via [Git LFS](https://git-lfs.com):

| File | Contents |
|------|----------|
| `checkpoint/weights.pt` | Monroe encoder weights (~300 MB, LFS) |
| `checkpoint/config.json` | Architecture config the loader reads |

## Citation

If you use this code, please cite:

```bibtex
@misc{banaszewski2026monroe,
      title={Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference}, 
      author={Blazej Banaszewski and Andrew W. Fitzgibbon},
      year={2026},
      eprint={2608.18982},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.18982}, 
}
```

## License

Released under the [MIT License](LICENSE).
