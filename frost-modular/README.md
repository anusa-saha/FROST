# FROST Modular

This folder turns the notebook implementation into a small CLI-oriented codebase.
It keeps the paper-aligned token shortlist decoder, uses K-FAC-only proxy scoring, and adds reusable evaluation and plotting utilities.

## Layout

- `cli.py` - thin launcher so you can run `python cli.py ...`
- `model_zoo.json` - model registry used by CLI keys
- `requirements.txt` - runtime dependencies
- `frost_modular/` - reusable package

## Install

```powershell
cd frost-modular
pip install -r requirements.txt
```

## Model Zoo

The CLI resolves models by key from `model_zoo.json`.

Default keys:

- Teacher: `deepseek_r1_qwen7b`
- Proxy: `mistral_7b_instruct_v03`

You can edit `model_zoo.json` to add more teachers or proxy students without changing the code.

## CLI Commands

Run all commands from inside `frost-modular/`.

### Decode one prompt

```powershell
python cli.py decode --prompt "Question: 2+2?\nAnswer:" --beta 0.5 --shortlist-k 5
```

Useful arguments:

- `--model-zoo`: path to the registry JSON
- `--teacher-key`: teacher key from the zoo
- `--proxy-keys`: one or more proxy keys from the zoo
- `--dtype`: override dtype for loaded models (`float16`, `bfloat16`, `float32`)
- `--device-map`: override device map, usually `auto`
- `--max-sequence-length`: tokenizer truncation length
- `--max-new-tokens`: maximum decoded tokens
- `--shortlist-k`: shortlist size for defended decoding
- `--beta`: FROST inverse temperature
- `--sample`: sample from shortlist instead of greedy selection
- `--kfac-rank`: projected K-FAC rank
- `--kfac-damping`: inverse-factor damping
- `--kfac-layer-limit`: how many linear layers to include in scoring
- `--calibration-samples`: number of GSM8K examples used for K-FAC calibration
- `--prompt`: prompt text
- `--prompt-file`: read prompt text from a file
- `--output-json`: optional JSON output path

### Evaluate on GSM8K

```powershell
python cli.py evaluate --eval-limit 8 --beta 0.5
```

Useful arguments:

- `--eval-limit`: how many GSM8K rows to evaluate
- `--beta-sweep`: optional list of beta values for a tradeoff curve
- `--output-json`: choose the JSON path, default is `gsm8k_frost_results.json`
- `--plot-dir`: where PNG graphs are written

The evaluation JSON includes:

- teacher and FROST outputs
- correctness labels
- lengths
- latencies
- mean candidate energy
- step-level diagnostic records

### Plot an existing results file

```powershell
python cli.py plot --results-json gsm8k_frost_results.json --plot-dir plots
```

Generated plots:

- accuracy / length tradeoff
- ROC-AUC curve
- threshold sweep with F1
- latency comparison
- beta tradeoff curve when multiple beta values are present

## Related CLI Arguments

Shared across commands:

- `--model-zoo`
- `--teacher-key`
- `--proxy-keys`
- `--dtype`
- `--device-map`
- `--max-sequence-length`
- `--max-new-tokens`
- `--shortlist-k`
- `--beta`
- `--sample`
- `--kfac-rank`
- `--kfac-damping`
- `--kfac-layer-limit`
- `--calibration-samples`
- `--output-json`
- `--plot-dir`

Decode-only:

- `--prompt`
- `--prompt-file`

Evaluate-only:

- `--eval-limit`
- `--beta-sweep`

Plot-only:

- `--results-json`

## Notes

- K-FAC is used as a decode-time curvature approximation for proxy learnability.
- Structural and provenance terms are present as stubs and default to zero.
- The code is designed to match the logic of `FrostDecoding.ipynb` but in reusable modules.
- The evaluation command writes `gsm8k_frost_results.json` by default unless you override `--output-json`.
