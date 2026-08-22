# Logbook

Dated experimental records, one file per session: `YYYY-MM-DD-slug.md`.

This is a lab notebook, not documentation. Entries are append-only and are not
rewritten when later findings supersede them — a wrong measurement that was
later corrected is part of the record, and the correction cites the entry it
overturns. Documentation describes what is true now; the logbook describes what
was observed, when, and on what apparatus.

Every entry states:

* **Apparatus** — exact hardware, versions, image digest, commit — *before* any
  observation, so a number can always be traced to the configuration that
  produced it.
* **Hypotheses, with the reasoning behind them, written before the result.**
  A hypothesis without a stated "why" cannot be wrong in an interesting way;
  when it fails you learn nothing about which belief to revise. Each is marked
  CONFIRMED, REFUTED, or PARTIAL against evidence.
* **Observations**, labelled measured / estimated / third-party. Never mix them
  unlabelled.

Conclusions that change the project's direction graduate into
`docs/decisions/` as an ADR. The logbook entry remains as the evidence it
cites.
