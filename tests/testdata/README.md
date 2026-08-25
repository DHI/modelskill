# Test data provenance

Most files here were produced for modelskill. The exceptions are listed below.

## From DHI/mikeio1d

These network files come from
[DHI/mikeio1d](https://github.com/DHI/mikeio1d/tree/main/tests/testdata)
(commit `d937466`), copied unchanged. mikeio1d is MIT-licensed, as is modelskill.

| File | Format | Used for |
|---|---|---|
| `network_cali.res11` | MIKE 11 | nothing here any more — see below |
| `epanet.res` | EPANET | nothing here any more — see below |
| `epanet.resx` | EPANET (MIKE+) | extra node quantities, merged onto the `.res` network |
| `epanet.inp` | EPANET input | real pipe lengths, which the `.res` does not carry |
| `swmm.out` | SWMM | nothing here any more — see below |

Reading these formats moved to mikeio1d with the rest of the topology layer
(ADR-013), and the tests that covered it moved with it. The files are kept because
mikeio1d has the same copies and modelskill may want EPANET-side coverage of its
own; nothing in this repository reads them today except `network.res1d`.

`epanet.resx` and `epanet.inp` pair with `epanet.res`: same run, same IDs. The
`.resx` node and reach IDs are a strict subset of the `.res` ones, and the `.inp`
`[PIPES]` IDs cover every `.res` reach except the pump.

`swmm.out` is kept without its `.inp` on purpose: the refusal it used to pin is
mikeio1d's now, and the file is the fixture that refusal needs.
