# Experiment log

One row per run. The columns map directly onto Table 1 of the paper draft, so a
measurement goes from this log into the manuscript without being retyped.

Rules that keep this log worth having:

- Fill in the **prediction** before the run, not after. A log of predictions you
  made and then checked is evidence; a log of results is just data.
- Never delete a row. A configuration that failed is a finding, especially an
  out-of-memory on the phone tier.
- Record the git commit. A number you cannot reproduce is not a number.
- Latency never appears without memory next to it.

---

## Run index

| # | Date | Pair | Model | Arch | Device | Backend | Quant | C | TTFT (ms) | TPOT (ms) | Tokens/s | Peak RAM (GB) | J/token | Quality | Status | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | | olmoe-vs-qwen15 | qwen2.5-1.5b | dense | | hf | none | 1 | | | | | | | | |
| 2 | | olmoe-vs-qwen15 | olmoe-1b-7b | moe | | hf | none | 1 | | | | | | | | |
| 3 | | | | | | | | | | | | | | | | |

---

## Run 1

- **Date:**
- **Commit:**
- **Device:** (paste the output of `python -m src.cli device`)
- **Question this run answers:**
- **Prediction (written before running):** predicted `C*` from
  `python -m src.cli predict`, and which arm I expect to win at `C=1`, and by
  how much.

### Command

```bash
python -m src.cli bench --pair olmoe-vs-qwen15 --concurrency 1 --repeats 5
```

### Result

| Model | Arch | TTFT (ms) | TPOT (ms) | Tokens/s | Peak RAM (GB) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| | | | | | | |

### Did the prediction hold?

- Predicted:
- Measured:
- If it missed, which term of the model was wrong (bandwidth, capacity, expert
  reuse, kernel quality)?

### What this changes

- Does this move a number in the paper? Which table?
- What is the next run, and what will it rule out?

---

## Run 2

(copy the block above)

---

## Open questions

Things noticed but not chased. Revisit before writing the discussion section.

- [ ]
- [ ]

## Dead ends

Things tried that did not work, with enough detail that you do not retry them in
week 6. These often become a paragraph in the limitations section.

- [ ]
