# Test data provenance

Most files here were produced for modelskill. The exceptions are listed below.

## From DHI/mikeio1d

These network files come from
[DHI/mikeio1d](https://github.com/DHI/mikeio1d/tree/main/tests/testdata)
(commit `d937466`), copied unchanged. mikeio1d is MIT-licensed, as is modelskill.

| File | Format | Used for |
|---|---|---|
| `network_cali.res11` | MIKE 11 | `Network.from_mike` coverage for `.res11` |
| `epanet.res` | EPANET | `Network.from_epanet` coverage |
| `epanet.resx` | EPANET (MIKE+) | the `resx=` companion — extra node quantities merged onto the `.res` network |
| `epanet.inp` | EPANET input | the `inp=` companion — real pipe lengths, which the `.res` does not carry |
| `swmm.out` | SWMM | asserting `.out` is refused — its reach connectivity lives in a companion `.inp` we do not read yet (#689) |

`epanet.resx` and `epanet.inp` pair with `epanet.res`: same run, same IDs. The
`.resx` node and reach IDs are a strict subset of the `.res` ones, and the `.inp`
`[PIPES]` IDs cover every `.res` reach except the pump.

`swmm.out` is kept without its `.inp` on purpose. It pins the refusal, so the test
fails the day we add SWMM support or a future mikeio1d starts reporting reach
connectivity for it.
