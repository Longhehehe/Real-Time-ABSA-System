# Persistent project instructions

For every material task performed in this repository:

1. Update `docs/DATASET_CONSTRUCTION_PROTOCOL.md` with a new task-log entry.
2. Record the objective, inputs, method or code change, outputs, measured
   results, decisions, limitations, and next dependency.
3. Clearly distinguish planned procedures from procedures actually executed.
4. Re-render `docs/DATASET_CONSTRUCTION_PROTOCOL.docx` by running:

   ```powershell
   .\.venv\Scripts\python -X utf8 .\scripts\render_dataset_protocol.py
   ```

5. Validate that the DOCX can be converted back to plain text with Pandoc.

Never rewrite or delete `data/raw/` during dataset cleaning. New cleaned
datasets must be published as versioned releases with per-record decision
ledgers and checksums.
