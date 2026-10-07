"""CoRL 2026 RoboEval Diagnostic Bimanual Challenge tooling.

Three entry points, all run as modules:

    python -m roboeval.challenge.evaluate  # run a policy, write logs/ and results.json
    python -m roboeval.challenge.validate  # schema + consistency (+ replay) checks
    python -m roboeval.challenge.package   # checksum manifest and upload archive

See docs/challenge.md for the participant workflow.
"""
