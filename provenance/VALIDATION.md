# Release preparation validation

Prepared 2026-09-12. No new full experiment was trained.

- Original development and confirmation source fingerprints match their recorded protocols.
- All 864 development and 720 confirmation candidate evaluation files match their recorded SHA-256 digests.
- The filtered 864-row development bank matches the original validation fields and both recorded provenance hashes.
- The archived guarded selector exactly reproduces all 108 target decisions. Refitting with local NumPy reproduces the selected guard configuration and policy choices; tiny coefficient differences across BLAS environments are allowed.
- The standalone release verifier checks the full package manifest and these evidence links using Python's standard library. A deliberately altered manifest is rejected.
- Added Python utilities and original source pass syntax parsing. Local documentation links and notebook code cells were checked during packaging.
- The replay wrapper reports missing dependencies clearly and preserves archived evidence.

The packaging environment has Python 3.12 and NumPy but no PyTorch. The original full unit suites, smoke training, and full bootstrap report replay were not rerun in this environment. Archived experiment completion is documented by the original records; the packaging checks verify consistency and selector replay, not an independent replication of GPU training.

The manifest authenticates consistency with this package's bytes. It is not an independent source of experimental timestamps or a guarantee of scientific correctness.
