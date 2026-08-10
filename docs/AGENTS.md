# AI Assistant Operating Instructions & Guidelines — GestureFlow

This document defines how any AI assistant (Antigravity, Gemini, GPT, Claude, or other pair-programming agents) must operate when working on the GestureFlow codebase.

---

## 1. Project Overview & Scope

**GestureFlow** is defined strictly as follows:

> "GestureFlow is a CNN-based Hand Gesture Recognition System developed as a machine learning project using the LeapGestRecog dataset. The project demonstrates the complete machine learning workflow, culminating in a real-time desktop webcam application that performs live gesture recognition using the trained CNN model."

GestureFlow is **not** a production deployment, web application, backend service, or full-stack platform. All development efforts focus exclusively on end-to-end Machine Learning engineering rigor.

---

## 2. Core Machine Learning Lifecycle

The project strictly follows a 10-step sequential workflow:

```
Dataset Research → Dataset Audit → Preprocessing → Subject-aware Dataset Split → CNN Development → Training → Evaluation → Model Selection → Real-Time Webcam Inference → Documentation
```

---

## 3. Mandatory Development & Phase Workflow

Every task implementation **must** strictly execute the following sequential workflow:

1. **Read `docs/AGENTS.md`**: Understand assistant operational boundaries.
2. **Read `docs/memory.md`**: Identify current project state and recent decisions.
3. **Read `docs/phases.md`**: Identify the single `ACTIVE` phase (Phase 2 — Dataset Audit & Preprocessing).
4. **Review Design & Architecture**: Inspect `docs/architecture.md`, `docs/rules.md`, and `docs/design.md`.
5. **Verify Scope**: Continue **only** the tasks corresponding to the single active phase.
6. **Implement Incrementally**: Write clean, modular, production-grade code without over-engineering or adding web stack bloat.
7. **Verify Implementation**: Execute unit tests, verify syntax/linting, and check performance metrics.
8. **Update `docs/memory.md`**: Document completed tasks, current status, and next steps.
9. **Update `docs/changelog.md`**: Add semantic entry under `[Unreleased]`.
10. **Update `docs/decisions.md`**: Record an ADR if architectural or technical choices changed.
11. **Update `docs/phases.md`**: Mark current phase `COMPLETED` and activate the next phase **only** when all exit criteria are met.
12. **Conclude & Report**: Provide a concise summary to the user with verification evidence.

---

## 4. Documentation Workflow

- The files in `docs/` are the **Single Source of Truth** for GestureFlow.
- Code must reflect documentation specs; if code and docs diverge, report the discrepancy to the user before proceeding.
- Never write implementation code without keeping corresponding documentation updated.

---

## 5. Coding & Architecture Rules

- **Python Standards**: Follow PEP 8, enforce type annotations, use Google-style docstrings, format with `black`, lint with `flake8`.
- **Subject-Aware Splitting**: Enforce subject isolation (`00`–`09`) across splits ($70\% / 15\% / 15\%$). Random image-level splitting is permanently prohibited.
- **Directory Layout**: Segregate source code into `src/` subpackages: `config/`, `dataset/`, `models/`, `training/`, `evaluation/`, `inference/`, and `utils/`.
- **Desktop Webcam Inference**: OpenCV handles camera capture and prediction overlay displays. MediaPipe may be used **only** for hand ROI detection; gesture classification **must** be performed by the trained PyTorch CNN model.

---

## 6. Testing Workflow

- Automated test suites reside under `tests/`.
- Run pytest before marking any task as complete:
  ```bash
  pytest tests/
  ```
- Hardware devices (such as webcams) must be mocked during automated testing.

---

## 7. Definition of Done (DoD) Checklist

A feature or phase is **DONE** only when:
- [ ] Code meets all functional requirements in `docs/prd.md`.
- [ ] Code conforms to engineering standards in `docs/rules.md`.
- [ ] Subject-aware dataset splitting rules are strictly verified.
- [ ] Automated tests pass with zero errors.
- [ ] Artifacts (plots, JSON stats, model checkpoints, reports) are generated in appropriate `outputs/` or `models/` folders.
- [ ] `docs/memory.md`, `docs/changelog.md`, and `docs/decisions.md` are updated.
- [ ] Verification evidence is presented in the final summary.

---

## 8. Rules for Modifying Files

- **New Files**: Must be placed in the appropriate directory per `docs/architecture.md`.
- **Existing Files**: Modify incrementally; preserve existing comments, docstrings, and contracts unless refactoring is requested.
- **Deletions**: Never delete source or documentation files without user confirmation.

---

## 9. Communication Style

- Communicate like a senior ML software engineering lead.
- Keep responses concise, structured, and focused.
- Provide direct file links (`file:///path/to/file`) when referring to workspace files.
