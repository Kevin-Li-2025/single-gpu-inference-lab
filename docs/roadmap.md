# Roadmap

Single-GPU Inference Lab is an evidence-driven LLM inference systems workspace
for real single-card serving. The primary measurement target remains NVIDIA L20
48 GB, but the project should win by being reproducible, honest about limits,
and portable enough to use A100 controls when they sharpen a claim.

## Positioning

The flagship story:

> A reference stack for training, fine-tuning, serving, benchmarking, and publishing LLM experiments on one L20 GPU, with every claim tied to a config, command, and measured artifact.

The project should avoid broad framework claims until the implementation earns them. The first public narrative should be narrow:

- single-GPU reproducibility
- QLoRA-first fine-tuning
- memory planning before job launch
- benchmark reports that include latency, throughput, and peak memory
- model cards generated from experiment manifests

## Version Plan

### v0.1: Reproducible QLoRA Baseline

Goal: one real fine-tuning path that can run on a single L20 and be verified from config to output.

Deliverables:

- QLoRA training entry point with PEFT, Transformers, Accelerate, and bitsandbytes.
- Tiny checked-in JSONL fixture for smoke tests.
- Real dataset config path or Hugging Face dataset switch.
- Training report containing command, model, dataset, hyperparameters, peak memory, runtime, and final loss.
- Basic benchmark harness for tokens/sec and memory around the training loop.
- `make verify` or `python -m l20_stack.verify` for local checks.

Acceptance:

- Smoke training runs without downloading a large model.
- Real L20 run completes with recorded peak memory.
- Tests pass without CUDA.
- No raw dataset, checkpoint, token, or secret is committed.

### v0.2: Model Card and Experiment Matrix

Goal: make every run publishable and comparable.

Deliverables:

- Hugging Face model card generator from experiment manifest and result JSON.
- Preset matrix for train, eval, serve, and quantization configs.
- Result schema for metrics and hardware metadata.
- Experiment index under `runs/` or `outputs/`, ignored by Git, with exportable summaries.
- Documentation for reproducing a run from a clean checkout.

Acceptance:

- A model card can be generated without manual editing.
- Presets validate before launch.
- A completed run produces a machine-readable report and a Markdown summary.

### v0.3: Serving and Quantization Baselines

Goal: measure inference before optimizing it.

Deliverables:

- vLLM baseline serving command.
- Request fixture format for prompt length, output length, batch shape, and concurrency.
- Benchmark harness for prefill latency, decode tokens/sec, p50/p95 latency, and peak memory.
- Quantization preset comparison.
- Report template for repeated runs on the same hardware.

Acceptance:

- One model has a serving baseline on L20.
- Benchmark results include raw JSON and rendered Markdown.
- Any optimization proposal points to a measured bottleneck.

### v1.0: Publishable Domain Model

Goal: use the stack to produce one real, documented model artifact.

Candidate directions:

- code assistant tuning
- finance-domain reasoning
- Chinese-English technical assistant

Deliverables:

- One released model or adapter.
- Model card with intended use, data summary, limitations, evals, and reproduction command.
- Training and serving reports.
- Baseline comparisons against the base model.
- Public-facing README that links code, model card, and benchmark report.

Acceptance:

- The model artifact can be loaded by a clean user.
- Evaluation claims are reproducible from checked-in configs.
- Known limitations are documented, including single-GPU constraints.

## Commit Plan

Each commit should be independently reviewable and keep tests green.

### v0.1 Commits

1. `Add verify command`
   - Add `src/l20_stack/verify.py`.
   - Run tests, compile checks, config validation, and secret pattern scan.
   - Acceptance: `PYTHONPATH=src /usr/bin/python3 -m l20_stack.verify` exits 0.

2. `Add training data fixture`
   - Add tiny JSONL instruction fixture under `tests/fixtures/`.
   - Add parser tests.
   - Acceptance: fixture loader returns deterministic examples.

3. `Add dataset loader abstraction`
   - Support local JSONL first.
   - Leave Hugging Face datasets as optional dependency path.
   - Acceptance: no network needed for tests.

4. `Add QLoRA training config schema`
   - Extend config validation for optimizer, scheduler, save strategy, and eval interval.
   - Acceptance: bad configs fail before training starts.

5. `Add smoke trainer interface`
   - Implement a dependency-light dry-run trainer that validates plumbing without Torch.
   - Acceptance: smoke trainer emits a result JSON.

6. `Add real QLoRA runner`
   - Add optional Torch/Transformers/PEFT path.
   - Keep import errors actionable.
   - Acceptance: CUDA-free tests still pass; L20 command is documented.

7. `Record training telemetry`
   - Capture runtime, tokens processed, memory if CUDA is available, and output paths.
   - Acceptance: result JSON has stable fields.

8. `Document first L20 runbook`
   - Add exact setup, command, expected outputs, and troubleshooting.
   - Acceptance: a clean user can reproduce the smoke path.

### v0.2 Commits

9. `Add result schema`
   - Define typed result objects for train, eval, serve, and environment metadata.
   - Acceptance: schema round-trips JSON.

10. `Add model card generator`
    - Generate Markdown from config and result JSON.
    - Include data, metrics, limitations, and reproduction command.
    - Acceptance: generated card is deterministic in tests.

11. `Add preset matrix`
    - Add named presets for small smoke, L20 QLoRA, eval-only, serve-only, and quantized serve.
    - Acceptance: all presets validate in verify.

12. `Add run export command`
    - Convert result JSON into Markdown report.
    - Acceptance: report includes config hash and command.

### v0.3 Commits

13. `Add serving request fixtures`
    - Define prompt/output length and concurrency fixtures.
    - Acceptance: fixtures validate without vLLM.

14. `Add vLLM baseline command`
    - Implement optional vLLM path with clear missing-dependency errors.
    - Acceptance: dry-run mode prints launch plan.

15. `Add inference benchmark runner`
    - Record latency, tokens/sec, request shape, and memory.
    - Acceptance: benchmark output conforms to result schema.

16. `Add quantization comparison presets`
    - Add baseline, int8, int4, and engine-specific preset files where supported.
    - Acceptance: unsupported combinations fail with clear messages.

17. `Add benchmark report renderer`
    - Render repeated-run summaries with p50/p95 and variance.
    - Acceptance: deterministic report from fixture results.

### v1.0 Commits

18. `Select domain model target`
    - Add a short decision record for data, base model, evals, and release risk.
    - Acceptance: one target is chosen; alternatives are explicitly deferred.

19. `Add domain dataset recipe`
    - Implement dataset preparation without committing raw data.
    - Acceptance: recipe validates paths and produces stats.

20. `Add evaluation suite`
    - Add task-specific eval config and baseline comparison.
    - Acceptance: eval report compares base model and tuned adapter.

21. `Train and package adapter`
    - Produce adapter artifact outside Git.
    - Acceptance: packaging manifest records checksums and load command.

22. `Generate release model card`
    - Generate final model card from real run outputs.
    - Acceptance: card includes limitations and exact reproduction command.

23. `Polish public README`
    - Make first screen explain the stack, hardware, quickstart, and evidence links.
    - Acceptance: README links to model card, runbook, benchmark report, and roadmap.

## Verification Gates

Every commit:

```bash
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/private/tmp/single-gpu-infer-pycache PYTHONPATH=src /usr/bin/python3 -m compileall -q src tests
git diff --check
```

Before every push:

```bash
grep -RInE "(gho_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY)" . --exclude-dir=.git --exclude-dir=__pycache__
```

Before any public claim:

- Re-run the command from a clean checkout.
- Record hardware, driver, CUDA, Python, package versions, and config hash.
- Keep raw outputs outside Git but export summaries into docs.
- Mark unmeasured performance as hypothesis, not result.
