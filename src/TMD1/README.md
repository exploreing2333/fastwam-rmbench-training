# TMD1

Frozen backup of the current TMD model path.

Entry points:

- `TMD1.runtime.create_tmd`
- `TMD1.fastwam_qwen.models.qwen3vl_mot.TMD`

Compatibility aliases are kept:

- `TMD1.runtime.create_fastwam_qwen3vl`
- `TMD1.fastwam_qwen.models.qwen3vl_mot.FastWAMQwen3VL`

The project-local dependencies used by this model were copied into this
package and rewritten to import from `TMD1.*`, so later edits to the top-level
`fastwam` or `fastwam_qwen` packages should not affect this backup. External
runtime assets such as the Qwen checkpoint, Wan/ActionDiT checkpoints, datasets,
and Python packages are still loaded from their configured paths/environment.
