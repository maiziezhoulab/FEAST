# Release notes

## 1.1.2

Modeled and transformed gene zero fractions now span 0 to 1, including rare genes above 99% zeros. Local-reference generation accepts `zero_target_source="modeled"` (the default) or `"observed"`. The latter calibrates the sampled counts to measured fractions in the eligible source groups using the configured reference weights and combination rule.

Calibration honors the rounded zero count first. An all-zero target produces zero counts. Otherwise the allocator reserves one count per requested positive. Reference count ceilings can further limit the output. Count diagnostics report the rounded requested total and the realized total minus that request. This release also includes the 1.1.1 positive-count redistribution correction.

## 1.1.1

Count calibration now distributes counts above the one-count minimum in proportion to the sampled excess at each positive spot. Integer counts that already meet the requested total and zero count retain their values. When every positive starts at one, calibration distributes any extra counts as evenly as possible.
