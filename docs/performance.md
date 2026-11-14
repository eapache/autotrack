# Performance measurements

The benchmark synthesizes an original repeating eight-note melody on “doo” at
TIGER's default style, seed and diffusion stride (20). Each duration is one
uncached inference chunk. This deliberately bypasses the normal 13-second
phrase grouping limit to measure longer contexts. Every configuration gets one
warm-up followed by two measured runs; reported times are their median. Session
initialization is recorded separately. Chunk timings include inference and a
cache WAV write, but exclude MP3 encoding and the initial model-file hashing.

Longer *songs* normally use more short chunks rather than one long context.
Increasing `--phrase-seconds` changes phrasing and memory use as well as speed;
a faster benchmark is not evidence of better singing quality.

Results are machine-specific. The development workstation has an AMD Ryzen 9
9900X (12 physical cores / 24 logical CPUs) and an NVIDIA RTX PRO 4500 Blackwell.
Measurements use ONNX Runtime 1.32.0. They were taken on a live desktop, not an
isolated performance lab; small differences should not drive defaults.

## Reproduce

Run the CPU benchmark with no other render running:

```sh
python scripts/benchmark.py --voice models/tiger \
  --threads 1 2 4 8 12 24 --seconds 12 --repeats 2 \
  --out output/cpu-threads.json
python scripts/benchmark.py --voice models/tiger \
  --threads 8 --seconds 4 12 30 60 --repeats 2 \
  --out output/contexts.json
```

The script clears the phrase cache before every attempt and records the
individual samples, settings, session times, and real-time factor (synthesis
seconds / audio seconds). Failures are recorded as errors and cause a nonzero
exit status.

## Historical CUDA evaluation

CUDA support was removed after this evaluation; the current CLI, renderer and
benchmark are CPU-only. The GPU measurements below and their raw records are
retained as historical evidence for that decision, not as supported setup instructions.

## Results

Wall-clock seconds (lower is better), at diffusion stride 20:

| CPU threads | 4 s audio | 12 s audio | 30 s audio | 60 s audio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.47 | 9.88 | 25.04 | 51.23 |
| 2 | 1.94 | 5.53 | 13.90 | 29.92 |
| 4 | 1.19 | 3.33 | 8.65 | 19.28 |
| 8 | 1.03 | 2.67 | 6.33 | 13.38 |
| 12 | 1.03 | 2.59 | 6.00 | 12.19 |
| 24 | 1.24 | 3.20 | 7.41 | 14.27 |

| CUDA (8 host threads) | 4 s audio | 12 s audio | 30 s audio | 60 s audio |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO 4500 | 0.49 | 3.59 | 21.28 | not completed |

CUDA helped at four seconds, but lost to CPU at twelve and thirty seconds.
The sixty-second CUDA case was stopped after the twelve- and thirty-second results showed
no benefit; it is deliberately not assigned an estimated timing. The thirty-second
case was about 3.4 times slower than eight-thread CPU. This does not establish
GPU performance for other models, hardware, driver versions or quality settings.

A separate CPU confirmation (one warm-up, three measured runs, with no concurrent
development tests) gave 2.74 s at eight threads, 2.66 s at twelve, and 3.21 s at
24 for the twelve-second passage. The default is therefore **CPU, capped at eight
available threads**: near the best measured short-phrase throughput while using
fewer threads. Use `--threads 12` on this workstation when prioritizing long-chunk
throughput. Simply using every logical CPU was slower. The phrase grouping default
remains 13 seconds; longer chunks gave only modest CPU throughput gains and
worsened CUDA throughput in these measurements.

We also tested ONNX Runtime's documented
[alternative convolution input padding](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#convolution-input-padding)
(`cudnn_conv1d_pad_to_nc1d=1`) on the twelve-second CUDA passage. The median was
3.63 s versus 3.59 s with the standard layout, so it was not adopted. The probe
used the same renderer and inputs, changing only this provider setting before
creating its sessions. No inference kernel or model-graph optimization is claimed.

See [raw measurements](benchmark-results.json), including individual samples,
runtime settings, model hashes in the confirmation run, and experiment notes.
The first CPU sweep had occasional light development checks running alongside it;
the separate confirmation supports the chosen default without relying on small
differences from that sweep.
