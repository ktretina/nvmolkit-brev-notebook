# ACS Fall 2026 GPU chemistry workshop

Use one conversational workspace to answer four bounded chemistry questions with
OpenClaw, hosted NVIDIA Nemotron, nvMolKit, and one NVIDIA L4. No local coding
setup is required.

## Before the workshop

Complete these steps before you arrive:

1. Create [one NVIDIA account](https://account.nvidia.com/), verify your email,
   and complete an NVIDIA Cloud Account if prompted.
2. Sign in to [NVIDIA Brev](https://brev.nvidia.com/). Complete its onboarding
   and make sure your organization has Brev credits or a payment method.
3. Open the [NVIDIA API-key page](https://build.nvidia.com/settings/api-keys).
   Generate and copy an API key. Complete phone verification if requested. The
   required lab uses
   [NVIDIA Nemotron 3 Super 120B-A12B](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b?nim=hosted).

Hosted prototype access can be rate-limited. Brev GPU compute is separate and billable.
Never paste the key into chat, a screenshot, or a file. Paste it only into the
Launchable setup field:

| Lab | Setup field |
| --- | --- |
| Optional notebook | `NVIDIA_API_KEY` |
| Required OpenClaw lab | `NVIDIA_INFERENCE_API_KEY` |

Use the same private API-key value in either field.

## Choose your lab

**Optional instructor-led companion:**
[open the guided notebook Launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3HJtJW3qHg4Dw1I3xt75BfpBmZW).
It presents the fixed analysis in notebook form. It is not required for the hands-on workshop.

**Required hands-on lab:**
[open the conversational OpenClaw Launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3Hlp4pHBlTTlfDxfH41KkGhTeCV).
It is designed to let Nemotron use a fixed runner while nvMolKit performs the
chemistry work on the L4.

As of August 11, 2026, public HTTP checks confirmed that both links load a Brev
web application. These public HTTP checks do not prove signed-in deployability
or a successful fresh setup. This page makes no live-readiness or timing claim.

## Launch the required lab

1. Use the default hardware. Confirm that the visible row shows one NVIDIA L4,
   x86-64, 4 CPUs, 16 GiB RAM, and 128 GiB disk. The row does not need to show `g6.xlarge`.
2. Enter the API key in `NVIDIA_INFERENCE_API_KEY`, then deploy.
3. Wait until setup is ready. Open **Open Chemistry Agent** and create one new session.
4. Paste the four prompts below in order into that same session. Wait for each
   answer before sending the next prompt.

## Four prompts

<!-- ACS_PROMPT:01-data-and-representation:BEGIN -->
~~~text
Question: What is in the fixed molecule library, and how is it represented for comparison?

Work only in `/sandbox/.openclaw/workspace`.
Before chemistry work, read `/sandbox/.openclaw/skills/nvmolkit-usage/SKILL.md` once.
Do not install software or use the network.
Do not edit any fixed file or run an alternate command.

Run this command exactly once:
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson data-and-representation

If the command fails, report the error and stop. Do not repair or retry.
Use only the returned results. Answer with these six headings in this order:
Question
What ran
Measured result
Meaning
Scientific limit
Image and download location

Use at most three measured facts. State that the data are a deterministic 256-record ChEMBL convenience sample, not representative chemical space. State that Morgan/Tanimoto conclusions depend on the radius-2, 1024-bit hashed fingerprint. Report real GPU execution, not acceleration or speedup. Say that the current bundle is in **Download Results** at `workshop/results.zip`.

End with this exact line:
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/01-inspection/library_preview.png
~~~
<!-- ACS_PROMPT:01-data-and-representation:END -->

<!-- ACS_PROMPT:02-relationships-and-groups:BEGIN -->
~~~text
Question: Which molecules are similar, and how does fused Butina group them?

Work only in `/sandbox/.openclaw/workspace`.
Do not install software or use the network.
Do not edit any fixed file or run an alternate command.

Run this command exactly once:
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson relationships-and-groups

If the command fails, report the error and stop. Do not repair or retry.
Use only the returned results. Answer with these six headings in this order:
Question
What ran
Measured result
Meaning
Scientific limit
Image and download location

Use at most three measured facts. State that cutoff `0.40` is Tanimoto distance. State that the result depends on the radius-2, 1024-bit hashed fingerprint and that similarity `1.0` does not prove molecular identity. Report real GPU execution, not acceleration or speedup. Say that the current bundle is in **Download Results** at `workshop/results.zip`.

End with this exact line:
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/04-clusters/cluster_sizes.png
~~~
<!-- ACS_PROMPT:02-relationships-and-groups:END -->

<!-- ACS_PROMPT:03-sampled-3d-geometry:BEGIN -->
~~~text
Question: What sampled 3D geometries were generated and optimized?

Work only in `/sandbox/.openclaw/workspace`.
Do not install software or use the network.
Do not edit any fixed file or run an alternate command.

Run this command exactly once:
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson sampled-3d-geometry

If the command fails, report the error and stop. Do not repair or retry.
Use only the returned results. Answer with these six headings in this order:
Question
What ran
Measured result
Meaning
Scientific limit
Image and download location

Use at most three measured facts. State that the deterministic selected molecules are not centroids, medoids, or globally optimal representatives. State that sampled conformers are not experimental structures and MMFF94 energies compare sampled conformers within one molecule only. Report real GPU execution, not acceleration or speedup. Say that the current bundle is in **Download Results** at `workshop/results.zip`.

End with this exact line:
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/06-mmff94/optimized_structures.png
~~~
<!-- ACS_PROMPT:03-sampled-3d-geometry:END -->

<!-- ACS_PROMPT:04-objective:BEGIN -->
~~~text
Question: Can a bounded agent improve the weakest-link diversity of a four-molecule panel?

Work only in `/sandbox/.openclaw/workspace`.
Do not install software or use the network.
Do not edit any fixed file or run an alternate command.

Run `objective-start` exactly once with this command:
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-start

If `terminal` is `true`, run zero objective-step commands. If the result is pending, find the maximum numeric `predicted_score` in the displayed actions. Select one displayed action tied at that maximum; this is the best predicted `D_min`. Substitute the exact returned `state_id` and `swap_id` in this template. Keep both substituted values single-quoted; a swap ID can contain `->`.

env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-step --state-id 'STATE_ID_FROM_MENU' --swap-id 'SWAP_ID_FROM_MENU'

After each result, stop immediately when `terminal` is `true`. Otherwise, repeat with the new displayed menu. Run at most three objective-step commands in total.

If a command fails, report the error and stop. Do not repair or retry.
Use only the returned results. Answer with these six headings in this order:
Question
What ran
Measured result
Meaning
Scientific limit
Image and download location

Use at most three measured facts: baseline `D_min`, final `D_min`, and their change. Put the baseline panel, limiting pair, accepted swap or swaps, and final panel under **What ran**. State that `D_min` is the weakest-link diversity score within eight fixed candidates. It is a structural-descriptor objective and does not demonstrate unrestricted autonomous design or biological performance. Report real GPU execution, not acceleration or speedup. Say that the full bundle is in **Download Results** at `workshop/results.zip`.

End with this exact line:
MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/07-objective/final_panel.png
~~~
<!-- ACS_PROMPT:04-objective:END -->

## Download your results

The four key PNG images appear in chat. Open **Download Results** for the complete files.
Then open `workshop/` and download `results.zip`. A path that starts with `/sandbox/` is
an agent filesystem path, not a browser URL.

The ZIP contains the fixed input and provenance plus the generated README,
CSV, SDF, JSON, and PNG files available at that point in the workflow.

## Finish and remove your environments

Stop every workshop environment you started as soon as the exercise ends.
After downloading your files, **Delete each environment when you are finished**.
Stopped storage can still incur a charge. Deletion is permanent, so download
your results first.

## Scientific limits

- The data are a deterministic 256-record ChEMBL convenience sample, not representative chemical space.
- The clustering cutoff `0.40` is Tanimoto distance.
- The deterministic selected molecules are not centroids, medoids, or globally optimal representatives.
- Morgan/Tanimoto conclusions depend on the radius-2, 1024-bit hashed fingerprint, and similarity `1.0` does not prove molecular identity.
- MMFF94 energies compare sampled conformers within one molecule only.
- `D_min` is the weakest-link diversity score within eight fixed candidates.
- The run demonstrates real GPU execution, not acceleration or speedup.

These computations describe molecular structures and sampled force-field
geometries. They do not establish identity, biological activity, binding,
efficacy, safety, synthetic feasibility, or clinical value.

## Official links

- [Workshop repository](https://github.com/ktretina/nvmolkit-brev-notebook)
- [NVIDIA Brev quickstart](https://docs.nvidia.com/brev/getting-started/quickstart)
- [Official NVIDIA account and API-key instructions](https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/integrations/nvidia-integrations.html)
- [Brev stop, storage, and deletion guidance](https://docs.nvidia.com/brev/concepts/gpu-instances)
