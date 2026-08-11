# ACS chemistry workshop

- Read the installed `nvmolkit-usage` skill before chemistry coding.
- Do not install or upgrade packages. The tested chemistry stack is already present.
- Run chemistry Python commands with:

  ```bash
  env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 <script.py>
  ```

- Use the fixed workshop dataset at
  `/sandbox/.openclaw/workspace/data/sample_molecules.csv`.
- Write user files to `/sandbox/.openclaw/workspace/outputs`.
- When a task makes files, include the Python source, a short README, and a
  `results.zip` download bundle. Create a PNG only when a plot helps answer the
  question.
- Treat fingerprints and molecular similarity as computational descriptors.
  They are not evidence of biological activity, binding, efficacy, or safety.
