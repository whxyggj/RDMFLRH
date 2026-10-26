# RDMFLRH

MATLAB implementation of **Robust Deep Matrix Factorization With Low-Rank and Hypergraph Learning for Multi-View Data Processing**, Baicheng Pan, Hangjun Che, Chenglu Li and Hongfei Li, *IEEE Transactions on Consumer Electronics*, 71(2), 4807-4818, 2025.

[Paper / DOI](https://doi.org/10.1109/TCE.2025.3525523)

This compact release runs the complete BBCSport dataset using the author's original deep matrix factorization code. It includes the data and helper functions needed to run independently.

## Requirements

- MATLAB with **Statistics and Machine Learning Toolbox** (`KDTreeSearcher`, `knnsearch`, `randsample`, and the existing statistics argument parser).
- No compiled MEX files, FLANN installation or full GSPbox installation is required.

## Quick start

Clone or download the repository, change MATLAB's current folder to its root, and run:

```matlab
myDMF_BBCsport
```

Or run from a terminal in the repository folder:

```sh
matlab -batch "myDMF_BBCsport"
```

The script sets its own helper paths and loads data relative to its location. The random seed is fixed to 0. The default is **one full training run**, as in the original BBCSport entry script; initialization and final clustering each use 20 k-means replicates. Set `replic = 10` in the script for ten complete training runs. A one-run standard deviation is zero and does not measure run-to-run variability.

## Dataset and default settings

`data/BBCSport.mat` is copied from the author's `LRDMF/dataset` directory. Its 544 samples and 5 classes match the BBCsport dataset described in the paper.

| Variable | Dimensions | Description |
| --- | --- | --- |
| `data{1}` | 3183 x 544 | First view, samples in columns |
| `data{2}` | 3203 x 544 | Second view, samples in columns |
| `truelabel{1}` | 1 x 544 | Reference labels |

Reference labels are used for evaluation and the number of clusters. Each sample in each view is normalized to unit Euclidean norm inside `myDMF`.

| Setting | Value |
| --- | --- |
| Layers | `[50, 5]` |
| Low-rank penalty `lambda` | 0.1 |
| Hypergraph penalty `gamma` | 100 |
| Hypergraph neighbors | 5 |
| Maximum iterations | 1000 |
| Relative stopping tolerance | 1e-5 |
| Complete training runs | 1 |
| K-means replicates | 20 |

The default layers and parameters are retained from the original `myDMF_BBCsport.m`. Other existing layer choices are `[100, 5]` and `[100, 50, 5]`.

## Output

The script reports ACC, NMI, ARI, iteration count and training time. Results are saved to `results/BBCSport_result.mat` (ignored by Git):

- `res_record`: 7 x number-of-runs matrix; rows are ACC, NMI, purity, F-score, precision, recall and ARI.
- `V_star_record`: consensus representations, one 5 x 544 matrix per run.
- `labels_record`: predicted cluster labels for each run.
- `obj_record`: the original implementation's stopping-statistic history for each run.
- `error_count_record`: the number of increases in that statistic, as counted by the original code.
- `elapsed_seconds`: training time per run, including graph construction and initialization, excluding final clustering.
- `AC`, `NMI`, `purity`, `Fscore`, `Precision`, `Recall`, `AR`: mean and standard deviation.
- Layer sizes, penalties, run count and iteration limit.

`obj` returned by `myDMF` is the original weighted-residual surrogate plus regularizers used by its stopping rule. It is not a direct evaluation of the paper's arc tangent objective. This release preserves that statistic and its stopping rule, together with the original optimization updates. The internal historical name `Cauthy_NMF_cost` is also retained; the sample weights used by the updates are `1 / (2*||e||*(1+||e||^2))`.

## Validation

The complete BBCSport demo was run using MATLAB R2024b Update 3 with Statistics and Machine Learning Toolbox, starting from MATLAB's default path. With seed 0 and the default settings, the single run completed in 140 iterations and returned ACC **0.8934**, NMI **0.7490** and ARI **0.7413**. Training took about 53 seconds on the validation machine; runtime depends on hardware. Representation dimensions, nonnegativity, finite histories and clustering metrics were checked. A second run using the original `LRDMF/myDMF.m` and the same seed matched both the consensus representation and stopping-statistic history within an absolute tolerance of 1e-12.

These are single-run demonstration results, not the paper's ten-run aggregate. Results may vary with MATLAB/toolbox versions and numerical libraries.

## Layout and provenance

```text
myDMF.m                  Original factorization and update implementation
myDMF_BBCsport.m         Ready-to-run BBCSport entry script
data/BBCSport.mat        Complete two-view dataset
third_party/tools/       Initialization, clustering, normalization and metrics
third_party/gspbox/      Required GSPbox 0.7.0 subset and its GPL license
citation.bib             Paper citation
```

The two main MATLAB files already published in this repository match the author's `LRDMF` directory. The missing helper functions and data are sourced from that directory. The broader local experiment collection was also searched for dependencies. This release adds relative paths, a fixed seed, dependency checks, saved results and output checks; it removes commented-out experiment code from the entry script. Source encodings are normalized to UTF-8 and trailing whitespace is removed. The global `warning('off')` in `myDMF` is removed so that warnings remain visible; the numerical update formulas are retained.

This is a single-dataset release. Other datasets, parameter sweeps, ablation experiments and plotting scripts are omitted. It does not reproduce every table or figure in the paper.

Third-party notices are retained. The bundled GSPbox subset, including the locally modified unnormalized hypergraph Laplacian, is GPL-3.0-or-later; see [its license](third_party/gspbox/LICENSE). Normalization credits H. Kasai; evaluation credits the multiview consensus graph clustering implementation; the bundled `litekmeans` retains its existing MathWorks attribution. No additional blanket license is assigned to other author-supplied code or data.

## Citation

Please cite the paper when using this code:

```bibtex
@ARTICLE{10820964,
  author={Pan, Baicheng and Che, Hangjun and Li, Chenglu and Li, Hongfei},
  journal={IEEE Transactions on Consumer Electronics},
  title={Robust Deep Matrix Factorization With Low-Rank and Hypergraph Learning for Multi-View Data Processing},
  year={2025},
  volume={71},
  number={2},
  pages={4807-4818},
  keywords={Matrix decomposition;Noise measurement;Data mining;Manifolds;Robustness;Data processing;Measurement uncertainty;Loss measurement;Feature extraction;Consumer electronics;Multi-view data processing;deep matrix factorization;robust matrix factorization;hypergraph learning},
  doi={10.1109/TCE.2025.3525523}
}
```

## 中文运行说明

安装 MATLAB 和 Statistics and Machine Learning Toolbox，将当前目录切换到仓库根目录，运行 `myDMF_BBCsport`。代码、BBCSport 完整数据集及所需辅助函数均已包含，无需手动配置路径。

默认进行一次完整训练；需要重复实验时，将入口脚本中的 `replic` 改为 `10`。运行结果保存到 `results/BBCSport_result.mat`。此版本保留原始算法更新公式和默认参数，仅整理一个数据集的核心运行流程。
