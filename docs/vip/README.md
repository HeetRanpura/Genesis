# GENESIS: VIP Design Experience deliverables

Deliverables for the Vertically Integrated Projects vertical (MPSTME, NMIMS), domain **AI & Data Systems**.
Semester IV (Design Experience I) and Semester V (Design Experience II). Everything here describes real work in
this repository. Fields marked **[to fill]** (faculty mentor, other members, meeting dates) are not recorded in
the repository and must be filled in by the team with real information.

## Semester IV: problem definition, research and team integration

| Rubric criterion (weight) | Evidence |
|---|---|
| Problem identification (15) | [`sem4/01_Problem_Definition_Report.docx`](sem4/01_Problem_Definition_Report.docx) |
| Literature review (20) | [`sem4/02_Literature_Review_and_Technology_Survey.docx`](sem4/02_Literature_Review_and_Technology_Survey.docx): 35 references, each checked online on 27 Oct 2026 |
| Team interaction (15) | [`sem4/04_Team_Roles_and_Working_Agreement.docx`](sem4/04_Team_Roles_and_Working_Agreement.docx); PRs #1-#4; meeting template |
| Documentation (15) | [`sem4/05_Design_Notebook_and_Meeting_Templates.docx`](sem4/05_Design_Notebook_and_Meeting_Templates.docx); `logs/SESSION_LOG.md`, `logs/AGENT_LOG.md`, `HANDOFF*.md` |
| Independent learning (15) | `TEACHER.md` (RL study guide); independent-learning log template |
| Concept development (10) | [`sem4/03_Initial_System_Concept_and_Feasibility.docx`](sem4/03_Initial_System_Concept_and_Feasibility.docx) |
| Communication (10) | [`sem4/06_Concept_Presentation.pptx`](sem4/06_Concept_Presentation.pptx) (14 slides, speaker notes) |

## Semester V: system development, prototype and collaboration

| Rubric criterion (weight) | Evidence |
|---|---|
| System design (20) | [`sem5/01_System_Design_Document.docx`](sem5/01_System_Design_Document.docx): requirements, architecture, interfaces, data, performance, risks |
| Prototype development (20) | The repository: engine, learner, recorder, viewer. One-command demo: `make demo` |
| Teamwork & participation (15) | PRs #1-#4, handoffs, RACI in the team document |
| Sub-team collaboration (10) | [`sem5/03_Subteam_Coordination_Report.docx`](sem5/03_Subteam_Coordination_Report.docx) |
| Documentation (15) | the documents above, code docstrings, `CLAUDE.md`, `docs/WORLD.md`, `docs/AGENTS.md`, `docs/VIEWER.md` |
| Communication (10) | [`sem5/04_Technical_Presentation.pptx`](sem5/04_Technical_Presentation.pptx) (16 slides, speaker notes); review record in the coordination report |
| Quality of work (10) | [`sem5/02_Test_and_Validation_Plan.docx`](sem5/02_Test_and_Validation_Plan.docx): 59 tests, CI (`.github/workflows/tests.yml`), coverage, preregistration, defect log |

## Diagrams and charts
`diagrams/`: problem context, subsystems, team structure, roadmap, code structure, verification levels, CI,
collaboration, risk matrix, viewer architecture, literature map, and three result charts. The reports also reuse
the nine system diagrams in `reports/stage2/pilot_preprint/diagrams/` (by Advik Divekar).

## Rebuilding
Edit the content in `build/`, then:

```bash
cd docs/vip/build && npm install
cd ../../.. && .venv/bin/python docs/vip/build/make_vip_diagrams.py && .venv/bin/python docs/vip/build/make_vip_charts.py
node docs/vip/build/build_sem4_reports.js && node docs/vip/build/build_sem5_reports.js && node docs/vip/build/build_decks.js
```

Diagrams need `rsvg-convert` (`brew install librsvg`). The decks use chart images, not native charts, because
Keynote does not render charts made by pptxgenjs. In Pages, a heading can occasionally land at the bottom of a page
(Pages ignores "keep with next"; Word honours it).
