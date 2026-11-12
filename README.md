# pumpfun-top5-forecast-01
[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/ulwan66/pumpfun-top5-forecast-01/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/ulwan66/pumpfun-top5-forecast-01/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
A Monte Carlo forecast of the five largest pump.fun coins by market cap at each answer read. Per-coin drift and volatility are estimated from the snapshot history, log-cap paths are simulated to every read, and a newcomer allowance fitted from rank churn is left unnamed for coins the history never ranked.
## Run it
```sh
mkdir -p out && MAKEFUN_INPUT_DIR=../../pumpfun-top5-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
---
<sub>Entered on [make.fun](https://make.fun/?ref=s3fge52m&utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
