# IFToMM benchmark results used as references

Results contributed to the Library of Computational Benchmark Problems of the
IFToMM Technical Committee for Multibody Dynamics
(<https://www.iftomm-multibody.org/benchmark/>), downloaded from it by the
author and given to the project on 20 October 2026 (task 5.2, decision D44).
They are the contributors' results, not the project's; each file's name
keeps the contributor and the number the library gives the result.
`tests/kernel/test_kernel_iftomm.cpp` reads them.

| File | Problem | Contributor (library result number) | Columns used | Changed from the file given |
|---|---|---|---|---|
| `bricard_tagliapietra_109.txt` | A04, rectangular Bricard mechanism | Luca Tagliapietra (109) | t [s], then x, y, z of point P2 [m], every 0.01 s | nothing |
| `bricard_gonzalez_72.txt` | A04 | Francisco González (72) | t, x, y, z of P2, every 0.01 s | nothing |
| `bricard_pankiewicz_160.txt` | A04 | E. L. Pankiewicz (160) | t, P2_x, P2_y, P2_z (header row), every 0.05 s | nothing |
| `flyball_gonzalez_117.txt` | A05, stiff flyball governor | Francisco González (117) | t [s], s [m]: the sleeve's position along the spindle, every 0.1 s, 9 digits | re-encoded from UTF-16 to ASCII, CRLF to LF |
| `flyball_gonzalez_116.txt` | A05 | Francisco González (116) | t, DZ [m]: the same, 6 digits | CRLF to LF |

The other columns, not used: Tagliapietra's fifth (at most 9.5e-7 in
magnitude, probably an energy change); González's fifth (probably an energy
change, from -6.9e-7 at 0.01 s to -7.9e-4 J at 10 s) and sixth and seventh
(at most 1.1e-15 and 1.0e-13, probably constraint residuals); Pankiewicz's E,
Ekin, Epot (the energies; E is -29.43 J, as printed, in every row) and gpos,
gvel (constraint residuals, at most 9.6e-11 and 1.0e-11). The meanings
marked "probably" are read from the values; the files do not state them.
