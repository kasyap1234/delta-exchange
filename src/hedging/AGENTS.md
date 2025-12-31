# HEDGING MODULE

## OVERVIEW
Auto-hedging system using correlation analysis to protect losing positions.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Hedge logic | src/hedging/hedge_manager.py | HedgedPosition, auto-protect losers |
| Correlation calc | src/hedging/correlation.py | Rolling correlation for hedge ratios |
| Funding monitor | src/hedging/funding_monitor.py | Track 8h funding rates |

## CONVENTIONS
- **Default hedge ratio:** 30% (configurable via HEDGE_RATIO)
- **Min correlation threshold:** 0.65 for valid hedge pairs
- **Hedge pairs mapping:** BTCUSD↔ETHUSD, SOLUSD→ETHUSD
- **Auto-protect triggers:** Position PnL < -2%
- **Hedge direction:** Opposite primary (long primary → short hedge)

## ANTI-PATTERNS
- **NEVER hedge without** correlation > 0.65
- **NEVER manually track positions** → use exchange positions + HedgedPosition records
- **NEVER skip net_exposure calculation** when managing hedges
- **NEVER allow hedge_ratio > 0.5** without explicit risk approval
