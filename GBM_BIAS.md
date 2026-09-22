# Why the GBM baseline lost money in 2024

Reproduced by [`analysis/gbm_bias.py`](analysis/gbm_bias.py).

## Why GBM lost in 2024 (mathematical mechanism)

Not an outlier in the statistical sense -- a systematic model bias meeting a year that
punished it.

**The bias.** Both models consume the SAME fitted smile; GBM collapses it to one number,
the at-the-money vol, and prices every bucket from a single lognormal. On 2024-11-19 the
fitted smile runs 28.1% at K=5000, 24.7% at 5200, 21.2% at 5400, 17.8% at 5600, 14.3% at
5800, against an ATM vol of 11.9%. GBM uses 11.9% for all of them. A lognormal that narrow
concentrates probability in the region within about one standard deviation of spot, which
is exactly where the buckets sat:

    bucket (2024-11-19)   rel. to spot   GBM(ATM)   GBM(local IV)   BL(full smile)
    5600-5799.99             -3.6%        20.0c        14.3c           10.5c
    5400-5599.99             -7.0%         6.0c         6.9c            4.1c
    5000-5199.99            -13.7%         0.0c         2.1c            0.9c

Pricing each bucket edge at its own implied vol closes most of the gap, so the error is
mostly the LEVEL (one vol for every strike); the remainder is the smile-slope term, which
enters the Breeden-Litzenberger density through dsigma/dK and suppresses density below the
forward where the skew is steep. GBM has no such term.

**It is systematic, not a 2024 accident.** Mean GBM minus BL by bucket position, all
years, in cents: -5..-2% below spot +4.67, -10..-5% +4.34, -20..-10% +2.00, at spot +0.80,
+2..+5% above -3.06, >+5% -2.86. Per year for buckets 2-10% below spot: 2022 +3.35,
2023 +5.23, 2024 +4.67.

**Why it only bit in 2024.** SPX rallied from 4743 to 5917, so 82% of bucket-days sat BELOW
spot that year, against 58-59% in 2022-23. GBM's below-spot overpricing was therefore
applied to most of the board, and it reads overpricing as cheapness: 14 of its 16 buys were
below-spot buckets, and those buys lost $4.72 of a $5.20 net loss. SPX then settled at
5881.63, above EVERY bucket, so every long expired worthless.

**It is not a forecasting failure in general.** Brier scores are close (2024: BL 5.49 vs GBM
5.88 per thousand; 2022 GBM is actually better, 63.5 vs 66.5). The damage is specific to
which trades the bias generated, not to broad calibration.
