## Quick Start

1. Download benchmark datasets:

```bash
cd OpenCodeEval/data
bash dataset.sh
```

2. Install dependencies:

```bash
pip install -e .
```

3. **Configure Evaluation Scripts**  
   - Replace placeholders in the evaluation scripts with the actual model name and path.  
   - Adjust any other necessary settings (e.g., evaluation parameters, output paths) to suit your requirements.

4. Execute the evaluation script for your desired benchmark. For example, to evaluate using the `test-humaneval-ckpt-list.sh` script:

```bash
bash test-humaneval-ckpt-list.sh
```

   > **Note**: Ensure that all configurations are correctly set before running the script to avoid errors.
