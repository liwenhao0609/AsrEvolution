# Bacterial Sulfate Assimilation: Profiles, Gene Trees, and Reconciliation Results

This directory contains the profile hidden Markov models (HMMs), maximum-likelihood gene trees, and species-tree reconciliation summaries used to investigate the distribution and evolutionary history of bacterial sulfate assimilation genes.

## Repository structure

```text
.
├── README.md
├── run_anvio.sh
├── Asr_hmmprofile/
│   └── <gene>.hmm
├── Asr_phylogenesis/
│   └── <gene>.treefile
└── Asr_reconciliation/
    └── <species_tree>_<gene>_totalSpeciesEventCounts.txt
```

- `run_anvio.sh` runs the anvi'o metabolic-module reconstruction workflow.
- `Asr_hmmprofile/` contains 11 profile HMMs used for homologue searches.
- `Asr_phylogenesis/` contains 11 curated maximum-likelihood gene trees.
- `Asr_reconciliation/` contains 33 reconciliation summaries: 11 gene families analysed against each of three species chronograms.

## Gene families

Eleven gene families are represented:

| Gene | General role in sulfate assimilation |
|---|---|
| `sat` | Sulfate adenylyltransferase |
| `cysD` | Sulfate adenylyltransferase subunit 2 |
| `cysN` | Sulfate adenylyltransferase subunit 1 |
| `cysNC` | Bifunctional CysN–CysC enzyme |
| `cysC` | Adenylylsulfate kinase |
| `cysH` | Phosphoadenosine phosphosulfate reductase |
| `cysI` | NADPH-dependent sulfite reductase hemoprotein component |
| `cysJ` | NADPH-dependent sulfite reductase flavoprotein component |
| `sir` | Ferredoxin-dependent sulfite reductase |
| `cysK` | Cysteine synthase A |
| `cysM` | Cysteine synthase B |

Files follow three naming conventions:

```text
Asr_hmmprofile/<gene>.hmm
Asr_phylogenesis/<gene>.treefile
Asr_reconciliation/<species_tree>_<gene>_totalSpeciesEventCounts.txt
```

- `Asr_hmmprofile/*.hmm`: profile HMMs used to identify candidate homologues.
- `Asr_phylogenesis/*.treefile`: maximum-likelihood trees for the curated gene families.
- `Asr_reconciliation/*_totalSpeciesEventCounts.txt`: per-species-tree event summaries produced by AleRax reconciliation.

The three bacterial species chronograms used for reconciliation were obtained from Adrián A. Davín *et al*., *A geological timescale for bacterial evolution and oxygen adaptation*. The reconciliation prefixes identify the alternative molecular-clock chronogram used in each analysis:

| Prefix | Species chronogram |
|---|---|
| `ugam` | `Figure3_timetree.tre` |
| `ln` | `FigureS19c_timetree.tre` |
| `wn` | `FigureS19d_timetree.tre` |

## 1. Genome collection and quality filtering

Bacterial species-representative genomes and metadata were obtained from [GTDB release 226](https://data.ace.uq.edu.au/public/gtdb/data/releases/release226/226.0/). The principal inputs were:

- `gtdb_proteins_aa_reps_r226.tar.gz`
- `bac120_metadata_r226.tsv`

The analysis retained GTDB bacterial representative genomes with CheckM2 completeness of at least 90% and contamination below 5%. Genomes failing the GUNC chimerism screen were removed. To reduce effects from sparsely sampled phyla, phyla represented by fewer than two orders or fewer than 25 species were excluded.

```r
library(data.table)
library(dplyr)
library(tidyr)

bac <- fread("bac120_metadata_r226.tsv") |>
  filter(gtdb_representative == "t",
         checkm2_completeness >= 90,
         checkm2_contamination < 5) |>
  separate(gtdb_taxonomy,
           into = c("domain", "phylum", "class", "order", "family", "genus", "species"),
           sep = ";")

gunc <- fread("GUNC_226.tsv")
gunc$genome <- sub("_protein$", "", gunc$genome)
bac <- merge(bac, gunc[, .(genome, pass.GUNC)],
             by.x = "accession", by.y = "genome", all.x = TRUE) |>
  filter(pass.GUNC == "TRUE") |>
  group_by(phylum) |>
  filter(n_distinct(order) >= 2, n() >= 25) |>
  ungroup()
```

This procedure retained 84,655 bacterial species representatives distributed across 54 GTDB phyla.

> **Reproducibility note:** the filtering code used in the analysis applies a completeness threshold of `>= 90%` and evaluates phyletic representation at the **order** level. These settings reproduce the reported set of 84,655 genomes. They should not be replaced with `> 95%` completeness or a class-level filter when reproducing this dataset.

### GUNC screening

Chimeric assemblies were assessed using [GUNC](https://github.com/grp-bork/gunc). The resulting `GUNC_226.tsv` table was joined to the GTDB metadata by genome accession, and only records with `pass.GUNC == TRUE` were retained. A typical GUNC v1.0.6 invocation for predicted proteins is:

```bash
gunc run \
  --gene_calls \
  --input_dir species_proteins/ \
  --file_suffix .faa \
  --db_file gunc_db_progenomes2.1.dmnd \
  --threads 100 \
  --out_dir gunc_output/
```

## 2. Oxygen-tolerance phenotype prediction

Oxygen-tolerance phenotypes were inferred with the [Oxygen_ML](https://github.com/444thLiao/Oxygen_ML) framework using its 40 selected KEGG Orthology features. The archived gradient-boosted decision-tree model was loaded directly. The logistic-regression model was refitted from the archived training matrix and phenotype labels using the parameters reported by the developers.

The reproduction procedure and the small differences from the archived predictions are documented in [Oxygen_ML issue #2](https://github.com/444thLiao/Oxygen_ML/issues/2). 

```bash
git clone https://github.com/444thLiao/Oxygen_ML.git
cd Oxygen_ML
git lfs pull
```

The input `test.tsv` is a genome-by-feature binary matrix. Its first column contains genome identifiers, and the remaining columns include all features listed in `top40.txt`.

```python
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

with open("top40.txt") as handle:
    top40 = [line.strip() for line in handle if line.strip()]

kegg_data = pd.read_csv("keggbin_reduced.tsv", sep="\t", index_col=0)
trait_data = pd.read_csv("NCBI_trait.tab", sep="\t", index_col=0)
common_ids = kegg_data.index.intersection(trait_data.index)

phenotype_to_binary = {
    "aerobic": 1,
    "obligate aerobic": 1,
    "anaerobic": 0,
    "facultative": 1,
    "microaerophilic": 1,
    "obligate anaerobic": 0,
}

X_train = kegg_data.loc[common_ids, top40]
y_train = np.array([
    phenotype_to_binary.get(value, np.nan)
    for value in trait_data.loc[common_ids, "metabolism"]
])
valid = ~np.isnan(y_train)
X_train = X_train.loc[valid]
y_train = y_train[valid].astype(int)

lr_model = LogisticRegression(
    penalty="l2",
    C=0.1,
    class_weight="balanced",
    n_jobs=-1,
    solver="liblinear",
)
lr_model.fit(X_train, y_train)

with open("lr_model.pkl", "wb") as handle:
    pickle.dump(lr_model, handle)

test = pd.read_csv("test.tsv", sep="\t", index_col=0)
test = test.loc[:, top40].apply(pd.to_numeric, errors="coerce").fillna(0)

gbdt_model = xgb.Booster()
gbdt_model.load_model("gbdt_20240709_top40.xgb")
gbdt_probability = gbdt_model.predict(xgb.DMatrix(test.values))
lr_probability = lr_model.predict_proba(test)[:, 1]
lr_prediction = lr_model.predict(test)

results = pd.DataFrame({
    "genome_id": test.index,
    "gbdt_prob": gbdt_probability,
    "lr_prob": lr_probability,
    "gbdt_predicted": np.where(gbdt_probability >= 0.5, "aerobe", "anaerobe"),
    "lr_predicted": np.where(lr_prediction == 1, "aerobe", "anaerobe"),
})
results.to_csv("prediction_results_with_LR.tsv", sep="\t", index=False)

agreement = (results["gbdt_predicted"] == results["lr_predicted"]).mean()
print(f"GBDT/LR agreement: {agreement:.2%}")
```

The logistic-regression classifications were used in downstream comparisons of aerobic and anaerobic bacterial lineages.

## 3. Metabolic-module annotation

Gene families were annotated with eggNOG-mapper v2.1.13 using DIAMOND, with a minimum query coverage of 50% and an e-value threshold of 1e-7. Metabolic modules were subsequently reconstructed with anvi'o v9. Per-genome module tables were combined while retaining a single header:

```bash
bash run_anvio.sh
awk 'NR == 1 || !/^module\t/' *_metabolism_modules.txt > all_modules_combined.tsv
```

## 4. Identification of sulfate-assimilation homologues

Reference sequences were retrieved from InterPro v109.0. Fragmentary records and entries annotated as probable or putative proteins were excluded. Reference sequences for each gene family were aligned separately with MAFFT v7.525 using the `--auto` option and converted into profile HMMs with `hmmbuild` under default settings in HMMER v3.4.

```bash
mkdir -p aligned Asr_hmmprofile

parallel -j 11 \
  'mafft --auto {} > aligned/{/.}.aln' \
  ::: *.fasta

for alignment in aligned/*.aln; do
  gene=$(basename "$alignment" .aln)
  hmmbuild "Asr_hmmprofile/${gene}.hmm" "$alignment"
done
```

Candidate homologues were retrieved from the proteins associated with the bacterial chronogram. A permissive search threshold was used at this stage so that candidates could subsequently be evaluated using domain composition, sequence length, phylogenetic placement, and genomic-neighborhood information. Conserved domain architectures were examined using Pfam v38.2.

```bash
hmmsearch --cpu 30 -E 0.01 \
  --tblout timetree_asr.out \
  asr.hmm timetree.fa
```

## 5. Gene-tree reconstruction

Candidate sequences that passed the family-specific filters were aligned with MAFFT v7.525 using the L-INS-i strategy, trimmed with trimAl v1.4, and used to construct an initial maximum-likelihood tree with IQ-TREE v3.0.1. The best-fitting substitution model was selected automatically with ModelFinder.

```bash
mafft --maxiterate 1000 --localpair ids.fa > ids.aln
trimal -in ids.aln -out ids.trim -gt 0.15
iqtree -s ids.trim -m MFP -B 1000 --boot-trees -T AUTO -nm 3000 --prefix ids
```

Final homologous groups were curated using phylogenetic placement and genomic-neighborhood support. Final trees were inferred using an expanded protein-model search space:

```bash
iqtree -s gene.trim \
  -m MFP \
  -mrate E,I,G,I+G,R \
  -madd C10,C20,C30,C40,C50,C60,EX2,EX3,EHO,UL2,UL3,EX_EHO,LG4M,LG4X,CF4,LG+C10,LG+C20,LG+C30,LG+C40,LG+C50,LG+C60 \
  -B 1000 --boot-trees -nm 3000 -T AUTO --prefix gene
```

The resulting trees are provided in `Asr_phylogenesis/` as `<gene>.treefile`.

## 6. Gene-tree/species-tree reconciliation

Gene trees were reconciled against three bacterial chronograms with [AleRax v1.4.1](https://github.com/BenoitMorel/AleRax). Reconciliations used 1,000 gene-tree samples, relative-dating constraints on horizontal transfers, and gene-family-specific missing-data fractions.

The following loop summarizes the commands used for all 11 gene families and all three species trees:

```bash
genes=(cysI cysJ sir cysD cysN cysNC sat cysH cysC cysK cysM)
declare -A species_trees=(
  [ugam]="Figure3_timetree.tre"
  [ln]="FigureS19c_timetree.tre"
  [wn]="FigureS19d_timetree.tre"
)

for model in ugam ln wn; do
  mkdir -p "output_reconciliation/${model}"
  for gene in "${genes[@]}"; do
    alerax \
      -f "data/mapping/${gene}.family" \
      -s "${species_trees[$model]}" \
      -p "output_reconciliation/${model}/${gene}" \
      --gene-tree-samples 1000 \
      --transfer-constraint RELDATED \
      --fraction-missing-file data/FractionMissing.txt
  done
done
```

For each reconciliation, `totalSpeciesEventCounts.txt` was retained in `Asr_reconciliation/` and renamed as:

```text
Asr_reconciliation/<species_tree>_<gene>_totalSpeciesEventCounts.txt
```

## Software used

The workflow requires the following principal software packages:

- GTDB release 226
- CheckM2
- GUNC v1.0.6
- treePL
- iTOL
- Oxygen_ML and its Python dependencies
- eggNOG-mapper v2.1.13
- DIAMOND
- anvi'o v9
- Cytoscape v3.10
- InterPro v109.0
- MAFFT v7.525
- HMMER v3.4
- Pfam v38.2
- trimAl v1.4
- IQ-TREE v3.0.1
- ModelFinder, as implemented in IQ-TREE v3.0.1
- AleRax v1.4.1
- GNU Parallel


Exact resource paths, thread counts, and scheduler settings should be adjusted for the local computing environment.
