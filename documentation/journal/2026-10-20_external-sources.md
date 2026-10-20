# Ideas from external sources

- **Dates:** 20 October 2026
- **Plan:** between Phases 4 and 5, on branch `phase-5`; decision D42
- **Commits:** see the index
- **Written:** at the time

## Goal

Before Phase 5 the author shared what they had found interesting: five
GitHub repositories, a numerical-methods course, a product site, a plot
catalogue, an article and five documents, and afterwards two more
repositories. The question: which ideas would complete the program better,
and where would they go in the plan?

## What was done

- **Every source read.** The seven repositories cloned shallowly outside the
  project and read, not run: READMEs, licences, documentation and their
  central code (TUM's corner characterisation, Yano's controllers and tests,
  JAOT's testing record, ampl-agents' rules, the course's agent appendix,
  Fast-F1's notes on accurate calculation, REA's agent and error contracts).
  The web sources read with the posts behind them (MOVEdot's correlation
  campaign). The five PDFs in full.
- **Checked against our own code first**, so that nothing present was
  proposed again: our plots already use a colour-blind-safe palette; our
  dynamics algorithms are cross-checked only at single states, not along a
  trajectory; our modes carry no energy distribution; our curves are tables
  or lines; there is no road generator; nothing compares the car with
  measured data.
- **The review written**
  (`documentation/review_2026-10-20/external_sources.md`): what each source
  is, what is useful, caveats, a verdict; seven themes several sources share;
  fourteen proposals ranked by value against cost, each placed in the plan.
- **The author adopted all of them**, and the plan was amended in the phases
  where each belongs (D42): two new checks in 5.1, conflicting constraints
  named in 5.2, ISO 8608 roads (7.2), hysteretic elements (7.6, 7.8), strict
  model files and a parameter-reach test (8.2, 8.6), hardpoint
  sensitivities (9.1), driver models (9.3), derived channels (9.5),
  correlation with measured data (9.7, new), real laps (10.1, 10.7),
  discipline for fitted models and an agent interface (13.5, 13.8), parameter
  sensitivities (13.9, new), Adams/Car property files (14.7, new) and the
  working rules in the repository (H.7, new). Estimates raised by about 100
  to 150 hours.
- **Local copies** of the repositories (about 480 MB) and the web pages in
  `C:\dev\references\`, outside the repository, with an index of each
  source's commit, date and licence.

## What went wrong, and how it was found

- **MDPI refused the article** (403 to every automated request, including
  the PDF link). Its metadata and abstract came from Crossref and OpenAlex;
  the review says it rests on the abstract only.
- **A claim corrected before it stood.** The first draft said Yano's asserts
  were "mostly in helper libraries"; counting them per file showed all 161 in
  component tests and none in the 41 controller and planner tests, and the
  text now says exactly that.
- **The plan's Windows path** in the appendix note lost its backslashes to a
  perl replacement (`\r` became a carriage return); found by dumping the
  line's bytes and rewritten with escaped characters. The rule stands: use
  the Edit tool or escapes for text with backslashes.

## Lessons

- A source is worth most where it disagrees with what we do or shows what we
  lack: two of the strongest proposals (manufactured trajectories, the
  parameter-reach test) came from a course and an optimisation platform, not
  from the vehicle-dynamics repositories.
- Read the tests, not only the README: the guide with the most algorithms
  had the least checking.

## For the book

- How outside work was weighed: read, compared with what exists, a verdict
  and a place in the plan, the author's decision recorded (D42).
- The statement on the use of AI the book needs (outline, Part X), as the
  Díaz Rivera thesis has one.
