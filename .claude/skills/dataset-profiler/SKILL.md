---
name: dataset-profiler
description: Profile and analyze a dataset before running GAFIME feature interaction mining. Use when the user wants to know if their data is compatible with GAFIME, estimate VRAM requirements, check for problematic columns, determine optimal batch size, or asks things like "will my data fit on GPU", "profile my dataset", "analyze my CSV/Parquet", "how many features do I have", or "is my data ready for GAFIME".
---

# Dataset Profiler

Analyze a user's dataset and determine if it's ready for GAFIME mining, estimate resource requirements, and flag potential issues.

## Instructions

1. Ask the user for their data file path (CSV or Parquet).

2. Run the profiling script:

   ```bash
   python .claude/skills/dataset-profiler/scripts/profile_dataset.py "<file_path>"
   ```

   Optional flags:
   - `--target <column_name>` — specify the target column
   - `--vram <GB>` — specify available VRAM (default: 6.0)

3. Read the JSON output. It reports:
   - Row count and feature count
   - Column dtypes and null percentages
   - Zero-variance columns (useless for mining)
   - High-cardinality categorical columns
   - Estimated memory footprint (RAM and VRAM)
   - Recommended batch size for streaming
   - Whether the dataset fits entirely in VRAM
   - Data quality warnings

4. Present findings in a clear summary:
   - Data size overview
   - Feature quality assessment
   - VRAM fit analysis
   - Actionable recommendations

5. If issues are found, provide specific remediation steps:
   - Drop zero-variance columns
   - Handle NaN values (fill or drop)
   - Encode categoricals before mining
   - Use `GafimeStreamer` if data doesn't fit in VRAM

## Example

**User says:** "I have a 500MB Parquet file with 200 features, will it fit on my 8GB GPU?"

**Actions:** Run `profile_dataset.py data.parquet --vram 8.0`, parse output.

**Result:** "Your dataset has 1.2M rows x 200 features. At float32, that's 915MB raw. With 8GB VRAM (6GB usable), you'll need to stream in batches of ~400K rows. Here's the setup: ..."
