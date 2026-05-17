# Livenix — Task Runner Protocol

> Mechanical protocol for running tasks from [TASKS.md](TASKS.md). Any model can act as the **Dispatcher**; the right model is spawned per task.

---

## Roles

| Role | Who | What they do |
|---|---|---|
| **Dispatcher** | Any model in the parent session (Sonnet by default) | Reads TASKS.md, picks next task, spawns Implementer + Verifier sub-agents, updates status |
| **Implementer** | Per-task model from the dispatch matrix (Haiku / Sonnet / Opus) | Reads task spec, produces deliverables, self-checks, commits |
| **Verifier** | Always Opus | Runs QA verification commands cold, reports pass/fail. **No rewriting**. |

---

## Per-task flow

```
1. Dispatcher reads TASKS.md → picks next "pending" task with no
   unmet dependencies. Reads its Model tag from the dispatch matrix.

2. Dispatcher creates a self-contained prompt for the Implementer:
   "Branch: feature/<task-id-slug>
    Task: <T-ID> — <title>
    Inputs: <list of files>
    Deliverables: <list of files/changes>
    Acceptance criteria: <verifiable list>
    Commit message pattern: <type>(<scope>): <subject>
    Co-Authored-By trailer required.
    Report back: branch name + commit SHA + one-paragraph summary."

3. Dispatcher: Agent(model=<task.Model>, prompt=<above>)
   → Implementer runs, commits, returns summary

4. Dispatcher reads TASKS.md QA verification commands for the task,
   spawns Verifier:
   Agent(model="opus", prompt=
   "Branch is at <commit SHA>. Run the following QA verification
    commands listed in TASKS.md for <T-ID>:
    <commands>
    For each criterion: pass or fail. List specific failures.
    Do NOT modify any files. Do NOT rewrite. Report only.")

5. If Verifier returns ALL PASS:
   - Dispatcher updates TASKS.md status table: pending → done,
     records branch + commit SHA
   - Dispatcher commits the status update on a `chore/status-<T-ID>`
     branch (or amends if status branch is shared)
   - Move to next task

6. If Verifier returns ANY FAIL:
   - Dispatcher re-spawns Implementer with the failure summary
     appended to the original prompt
   - Loop max 2 retries; after 2 fails, mark task "blocked",
     surface to human

7. Loop until all tasks done or "step-by-step" mode pauses for
   human confirmation.
```

---

## Branch model (locked: develop + short-lived feature branches)

```
main                        (stable, releases only — empty for now)
└── develop                 (integration — all in-progress work lives here)
     │
     └── feature/<t-id-slug>  (short-lived, ONE per task, deleted after merge)
```

### Per-task branch lifecycle

1. **Dispatcher** creates the branch off develop:
   `git checkout develop && git checkout -b feature/<t-id-slug>`
2. **Implementer** does its work, commits on the feature branch
3. **Dispatcher** verifies, then merges back to develop with `--no-ff`:
   `git checkout develop && git merge --no-ff feature/<t-id-slug> -m "Merge T<X.Y>: <subject>"`
4. **Dispatcher** deletes the feature branch:
   `git branch -d feature/<t-id-slug>`

After a task: `git branch` shows just `main` + `develop` + (at most) the next active task's feature branch.

### Why --no-ff for the merge

Each merge commit on develop documents "task T-X.Y landed at this point." If we used fast-forward, the merge commit vanishes and develop just shows a linear stream of cherry-picks — harder to read what was a discrete task.

### Branch naming convention

- `feature/<t-id-slug>` for task work (e.g. `feature/t2.3-patch-contrastive-head`)
- `fix/<short-desc>` for hotfixes that aren't tied to a numbered task
- `docs/<short-desc>` for doc-only changes

---

## Commit message conventions

- Conventional commits with scope: `type(scope): subject`
- Body: 1-3 lines explaining WHY and any non-obvious decisions
- Always end with: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`

(Note: Co-Authored-By trailer always shows Sonnet 4.6 by convention even
when the Implementer was Haiku — this is the SDK norm for Claude-assisted
commits. We can change this later if you prefer model-specific trailers.)

---

## Modes

### Step-by-step mode (default for unproven tasks)
- Dispatcher waits for "next" from human before running the next task
- Catches off-rails behavior early
- Recommended for the first 4-5 tasks of any new phase

### Autonomous batch mode (after pattern is proven)
- Dispatcher runs all pending tasks in dependency order until either:
  - All tasks done
  - 2 consecutive failures (auto-pause)
  - Cost ceiling reached (configurable)
- Status committed after each task so a session crash doesn't lose progress

---

## Self-contained Implementer prompt template

```
You are the Implementer for task <T-ID> in the Livenix project.

REPO ROOT: /Users/hamdev/livenix
CURRENT BRANCH: <integration branch>

Create branch: feature/<t-id-slug>

INPUTS TO READ (only these, do not explore beyond):
- <list from TASKS.md>

DELIVERABLES:
- <list from TASKS.md>

ACCEPTANCE CRITERIA (self-check before committing):
- <list from TASKS.md>

CONSTRAINTS:
- Do NOT touch /Users/hamdev/Silent-Face-Anti-Spoofing/
- Do NOT change pyproject.toml unless the task explicitly says so
- Do NOT modify TASKS.md or RUNNER_PROTOCOL.md
- Use conventional commits with scope + Co-Authored-By trailer

After all deliverables exist and acceptance criteria self-checked:
1. Stage only the files you created/modified for this task
2. Commit on feature/<t-id-slug> with the conventional commit message
3. Report back to dispatcher in this exact format:

   Branch: feature/<t-id-slug>
   Commit: <SHA>
   Summary: <one paragraph>
   Self-check: <list of acceptance criteria with PASS/FAIL>

If you hit a blocker, stop and report:
   STATUS: BLOCKED
   Reason: <one paragraph>
```

---

## Self-contained Verifier prompt template

```
You are the Verifier (Opus) for task <T-ID> in the Livenix project.

REPO ROOT: /Users/hamdev/livenix
BRANCH TO VERIFY: feature/<t-id-slug>
COMMIT TO VERIFY: <SHA>

Your job: run the QA verification commands listed below, observe outputs,
and report pass/fail per acceptance criterion. DO NOT modify any files.
DO NOT rewrite anything. Report only.

ACCEPTANCE CRITERIA:
- <list from TASKS.md>

QA VERIFICATION COMMANDS:
- <list from TASKS.md>

After running all commands, report in this exact format:

   Verifier: PASS  (or FAIL or PARTIAL)
   Per-criterion:
     - <criterion 1>: PASS / FAIL — <evidence>
     - <criterion 2>: PASS / FAIL — <evidence>
   Failures (if any): <list of specific issues>
   Recommendation: PROCEED / RETRY / BLOCK
```

---

## When to escalate to human

- Two consecutive Implementer failures on the same task
- Verifier returns "BLOCK" recommendation
- Any task touches a file outside its declared deliverables
- Any commit fails pre-commit hook
- Cost ceiling hit
- A task tagged manual (e.g. T2.13 dataset acquisition)
- Anything unexpected — the human is the safety net

---

## First-run checklist (for Dispatcher)

Before dispatching the first task in a session, verify:
- [ ] `git status` on the repo is clean (no uncommitted changes)
- [ ] Current branch is the agreed integration branch
- [ ] `uv run python -c "import livenix"` succeeds
- [ ] All previous tasks marked "done" in TASKS.md actually have commits in the log

If any check fails, surface to human before dispatching.
