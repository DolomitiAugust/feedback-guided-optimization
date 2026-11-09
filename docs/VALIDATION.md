# Publication checks

The curated public copy passed these checks without rerunning full policy training or online program search:

- 38 tests: nine LoRA/reward/data behavior tests, five independent recorded-evidence tests, six saved-search/geometry tests, 12 result-collection safeguards, two launcher-path checks, and four smoke-entry checks. The added checks use tiny local fixtures or mocks; they do not download a model or establish GPU performance.
- Training command configuration preview completed in dry-run mode, with no training started.
- A small CPU re-execution of the selected packing source produced a value within floating-point variation of the historical local result; the historical saved arrays and score were not overwritten.
- Saved Actor and Critic safetensors retained their original bytes. Only configuration paths were made portable.
- Review of the saved Critic header confirmed that its value head contains LoRA matrices without the underlying initialized weight. This limits independent Critic restoration and is disclosed in both reports; the verified Actor results are unaffected.
- The standalone search-framework plotting source regenerates a PNG pixel-identical to the published schematic. Saved-circle checks enforce the original evaluator's strict boundary and overlap rules, including rejection of tiny violations.
- Collection validates the model/run pairing before writing and refuses existing outputs. Its legacy records lack evaluation-time weight digests; no such historical digest was fabricated.
- Both reports were rebuilt from sanitized LaTeX and visually reviewed. Each has 14 pages; final compilation had no unresolved references or overfull boxes. Underfull reference formatting warnings remain.
- Published filenames, text, PDF content/metadata, image metadata and safetensors headers were scanned for the private identifiers and machine paths known to the source audit. No matches were found. No platform screenshots, executed notebooks, credentials or raw cloud archives are included.
- Relative Markdown links resolve locally. English and Chinese front pages link to the shared code and corresponding PDF preview/download.

Run `python tools/check_integrity.py` to verify the released content hashes. Running plotting scripts can regenerate files with different metadata; review the result before intentionally updating the manifest.

These checks establish the portable artifact's internal consistency. They do not turn a one-seed development experiment into a held-out benchmark, prove search convergence, or demonstrate a fresh clean-machine GPU installation.
