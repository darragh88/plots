# Bringing Main's Dependency Update Into Your Feature Branch

*(Assumes you have **already committed your local work** **and** run `git pull --ff-only` on `main` so it is up‑to‑date.)*  
*(Replace placeholders like `<SHA>` or `my-feature` with your actual branch name / commit IDs.)*

---

## Current Layout

```
(main)    A──B──C──D          ← origin/main & local main (D = env update)
(feature)           \──E──F  ← your branch (WIP commits)
```

Goal: add commit **D** (the dependency update) to your feature branch.

---

## OPTION 1 — Cherry‑pick only the env‑update commit

> Use when you just need **`environment.yml`** updated, not the rest of `main`.

```bash
# a. Identify the commit on main that updated environment.yml
git log main --oneline -n 10          # copy the SHA of that commit (e.g. d4e5c6a)

# b. Switch back to your feature branch
git switch my-feature                 # or: git checkout my-feature

# c. Cherry‑pick that commit onto your branch
git cherry-pick -x d4e5c6a            # "-x" appends origin info to the message

# d. Resolve conflicts if prompted (unlikely for a single-file change)
#    > If conflicts appear: edit file, git add <file>, git cherry-pick --continue

# e. Push the updated branch
git push                               # no force needed; history just grows

# f. Sync your environment
conda env update -f environment.yml --prune
```

**Pros:**  
* Minimal history noise—only the needed commit.  
**Cons:**  
* The same change will appear twice later when `main` is merged (harmless).

---

## OPTION 2 — Rebase your branch onto main (linear history)

> Good for solo branches where force‑push is acceptable.

```bash
git switch my-feature
git rebase main                       # replay E, F on top of D
# fix conflicts if any:
#   edit files → git add <file> → git rebase --continue
git push --force-with-lease           # update remote branch
conda env update -f environment.yml --prune
```

**Pros:**  
* Clean, straight history.  
**Cons:**  
* Requires `--force-with-lease` which can disrupt collaborators.

---

## OPTION 3 — Merge main into your branch (shared-safe)

> Use when teammates are also pushing to this branch or rebasing is disallowed.

```bash
git switch my-feature
git merge --no-ff main                # create a merge commit
# resolve conflicts if needed, then:
git push
conda env update -f environment.yml --prune
```

**Pros:**  
* No history rewrite, safe for shared branches.  
**Cons:**  
* Adds a merge bubble to the graph.

---

## Quick Decision Table

| Need | Pick |
|------|------|
| Only the env change, keep working | **Cherry‑pick** |
| Solo branch, want linear history | **Rebase** |
| Shared branch, avoid force pushes | **Merge** |

---

## After Any Option

1. **Verify the update**  
   ```bash
   git diff HEAD~1 environment.yml
   ```
2. **Run tests / start the app** to ensure new dependencies work.
3. **Continue coding** or open/refresh your PR.

_Last updated: 2025‑06‑20_
