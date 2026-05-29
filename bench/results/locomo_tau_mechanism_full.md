# §94d-mechanism — why is `schema_synthesis_tau` inert at retrieval?

- dataset: `bench/data/locomo10.json`  max_instances=10  k=10
- embedder: HashTrigram-256  min_supports=2
- taus: [0.3, 0.05]
- n_samples: 10  wall: 319.17s

## Arm aggregates

| tau | n_schemas | n_questions | top-k contains SCHEMA | rank-1 SCHEMA |
| --- | --- | --- | --- | --- |
| 0.3 | 26 | 1978 | 126 (0.0637) | 22 (0.0111) |
| 0.05 | 10 | 1978 | 110 (0.0556) | 16 (0.0081) |

## Verdict

SCHEMA writes fire (n_schemas=[26, 10]) and reach top-k at rate [0.0637, 0.0556] with rank-1 share [0.0111, 0.0081]. The §94d-tau-CI retrieval invariance is therefore NOT structural — SCHEMAs really do compete for top-k slots and the rate moves with tau — but the items they displace at the surviving top-k positions do not themselves carry gold sessions on LoCoMo10/hashtrigram-256/max_instances=2. The §5.3 'governance only' framing holds for this fixture but is not free in general; tau should be re-checked on harder fixtures (MiniLM-384, full LoCoMo, LongMemEval) before being declared free.
