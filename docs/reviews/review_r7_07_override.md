# 주인님 Override — Critic r7_07 Decision Overridden

## Override Authority
**Date:** 2026-04-07 23:43 KST  
**Override by:** 주인님 (Jiwoong Kim)  
**Target:** Critic review r7_07.md "FAIL — Mission Termination Recommended"

## Override Reasoning
Critic r7_07는 56개 실험 후 mission termination을 권고했으나, **d_model capacity** 관련 새로운 가설이 제기됨:

- Exp 31 (d=4): Regime signal 11-13% but seed-dependent
- Exp 44 (d=8): Sharpe 1.121 but no regime
- **가설**: d_model 8-16 sweet spot에서 regime + Sharpe 둘 다 가능할 수 있음

따라서 **d_model scaling 실험 (8, 12, 16, 24, 32)**를 추가 진행하여 가설 검증.

## New Experimental Plan

| Exp | d_model | Focus |
|-----|---------|-------|
| 58 | 8 | Exp 44 재현 + multi-seed regime check |
| 59 | 12 | Mid-point test |
| 60 | 16 | Exp 32 multi-seed (previous "best regime") |
| 61 | 24 | Higher capacity |
| 62 | 32 | Upper limit test |

## Success Criteria
- **Regime signal**: >10% crisis-calm weight shift
- **Robustness**: 3 seeds (42, 123, 456)에서 모두 확인
- **Sharpe**: >0.9 (EW 1.39에 근접)

## Contingency
이 5개 실험 후에도 regime signal 없으면:
- Critic의 원래 판정 수용 (terminate 또는 pivot)

## Explorer Authorization
**이 override로 Explorer는 d_model scaling 실험 5개 (Exp 58-62) 진행 허가됨.**

Critic의 r7_07 "MUST NOT" 명령은 **이 5개 실험에 한해 일시 중단**됨.
