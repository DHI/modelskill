# Test data provenance

Most files here were produced for modelskill. The exceptions are listed below.

## From DHI/mikeio1d

These network result files come from
[DHI/mikeio1d](https://github.com/DHI/mikeio1d/tree/main/tests/testdata)
(commit `d937466`), copied unchanged. mikeio1d is MIT-licensed, as is modelskill.

| File | Format | Used for |
|---|---|---|
| `network_cali.res11` | MIKE 11 | `Network.from_mike` coverage for `.res11` |
| `epanet.res` | EPANET | `Network.from_epanet` coverage |
| `epanet.resx` | EPANET (MIKE+) | asserting `.resx` is rejected — mikeio1d exposes no reach connectivity for it |
| `swmm.out` | SWMM | asserting `.out` is rejected — mikeio1d cannot resolve reach start/end nodes for it |

The last two exist to pin upstream behaviour: if a future mikeio1d exposes
connectivity for those formats, the rejection tests fail, which is when we would want
to add a constructor for them.
