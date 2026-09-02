# Archived 2026-05-06 — the migration checkpoints were never loaded

Eight runs, the first submission of the migration batch. They cannot be cited
for anything about migration.

## Why

Every run here that wrote metrics (7 of 8) opens at **KL from base ≈ 0.003** and
at the *untrained* perception loss for its substrate — 2.14–2.16 on Ashbourne
GC, 1.44–1.50 on Strathearn server. Those are the numbers of a fresh population.

The corrected re-runs one directory up open at KL 0.13–4.66 and perception loss
1.31–2.22, which is what resuming from an iteration-100 checkpoint looks like.

So these runs trained new populations under migration configs. The directory
name records the label the failure was given at the time (`kl0_bug`); what the
data shows is that the resume did not take effect.

The eighth run,
[`migration_callum_strathearn_server_to_ashbourne_gc/`](migration_callum_strathearn_server_to_ashbourne_gc/),
wrote no metrics at all. It is archived with its siblings because it was
submitted with them, not because it was diagnosed.

## What they could still be used for

Nothing has been re-derived from them. If anything, they are eight extra fresh
populations on the two Study Group framings at 150 iterations — but they were
never analysed as such, and the seed-replicate sets cover that ground properly.

## Never delete a run

These stay. The failure has to be findable, or the same submission gets made
again.
