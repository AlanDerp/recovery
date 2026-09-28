# PGSVP-v9 Cube Infrastructure Freeze

This freeze applies to the paper-grade `cube_stack` and `cube_restack`
A1/PGSVP paired experiments.

## Hard Boundary

Before starting A1, the paired runner verifies:

- every repository file listed in
  `experiments/pgsvp_v9_verify/infra_files.txt`;
- the Python executable, Python version, and complete installed-distribution
  metadata;
- SAM3 and Contact-GraspNet source commits;
- SAM3 and Contact-GraspNet checkpoint/config hashes;
- a clean repository and the recorded baseline-to-run patch hash;
- the canonical initial-response replay package when one is configured.

The computed fingerprint is saved as `INFRA_FINGERPRINT.json`. Any mismatch
stops the pair before an episode starts.

Verifier implementation, prompt, and schema are method artifacts. They are
hashed separately in `PAIR_MANIFEST.txt`, so method variants can be compared
without silently changing frozen infrastructure.

## Change Policy

After a valid freeze smoke, files in the infra list and the external service
revisions/checkpoints must not change for the associated experiment series.
An unavoidable infrastructure change requires:

1. a new lock and explicit infra version;
2. a fresh chain-health smoke;
3. rerunning A1 and all treatments under the new lock;
4. no comparison that mixes results from different infra locks.
