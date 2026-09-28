# Figure contract and QA

- Core conclusion: Task and Swap both emerge at 50% retention, but they diverge thereafter: Task remains comparatively high while Swap drops sharply.
- Figure role/archetype: validation; single-panel quantitative comparison.
- Backend: Python only (Matplotlib), including all exports.
- Final canvas: 89 × 67 mm before insertion into LaTeX.
- Primary format: SVG with editable text; companion PDF, 600 dpi TIFF, and 300 dpi PNG.
- Evidence mapping: the two grouped-bar series show Task and Swap independently; matching lines expose each series' trend across retention ratios.
- Sampling geometry: every Task and Swap line marker uses the exact x coordinate at the geometric midpoint of its corresponding bar.
- Statistics: descriptive observed proportions only. Each bar uses 500 trials; each aggregate uses 1,000 trials.
- Variability: no error bars are shown because per-task/per-seed observations were not supplied; treating all trials as independent binomial replicates could overstate precision.
- Accessibility: Swap is hatched and uses square markers, Task uses circular markers, and every value is numerically labeled, so the encoding does not rely on color alone.
- Source-data checks: all percentages reproduce the supplied counts; Task + Swap equals Long at every retention ratio.
- Aggregate audit across all ratios: Task 545/2,500 (21.8%); Swap 187/2,500 (7.48%); Long 732/5,000 (14.64%).
