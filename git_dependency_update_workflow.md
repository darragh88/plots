# Git Workflow: Importing Dependency Updates Into Your Feature Branch

*(A concise, repeat‑able guide you can paste into your team wiki or keep beside your IDE)*

---

## 0. What this covers

* **Scenario** – Someone updated **`environment.yml`** (or another dependency file) on **`main`**.
* **Goal** – Pull *just* that update (option A: **cherry‑pick**) **or** pull *all* of `main` (option B: **rebase / merge**) into **your current feature branch** while keeping your local changes safe.

Everything below can be run in a terminal or through PyCharm’s Git UI; GUI equivalents are listed where helpful.

---

## 1. Quick cheat‑sheet  📝

```bash
git status -sb                    # see branch + dirty files
git add -A && git commit -m "WIP" # or: git stash push -u

git fetch origin                  # update remote refs
git switch main && git pull --ff-only

git log -n 10 --oneline           # copy commit SHA that changed environment.yml
git switch -                      # jump back to previous branch
git cherry-pick -x <SHA>          # OR: git rebase main  /  git merge --no-ff main
git stash pop                     # if you stashed
conda env update -f environment.yml --prune
```

---

## 2. Detailed step‑by‑step

| # | Purpose | Terminal commands | PyCharm GUI |
|---|---------|------------------|-------------|
| 1 | **Check where you are & what’s changed** | `git status -sb`<br>`git diff` | *Git → Branches* (current branch is bold)<br>*Git → Show Diff* |
| 2 | **Save local work** | **Commit** →<br>`git add -A`<br>`git commit -m "WIP"`<br><br>**or Stash** →<br>`git stash push -u -m "before-main-update"` | *Commit…* or *Git → Stash Changes* (check **Include untracked** for blue files) |
| 3 | **Sync local `main`** | `git switch main`<br>`git pull --ff-only` | Checkout **main**, then *Pull* |
| 4 | **Find the dependency‑update commit** | `git log --oneline -n 10`<br>(look for _“update environment.yml”_) | *Git → Show History* on **main** |
| 5 | **Return to your branch** | `git switch my‑branch` | Checkout branch from list |
| 6A | **OPTION A – only that commit (cherry‑pick)** | `git cherry-pick -x <SHA>` | *Cherry‑pick* action in log |
| 6B | **OPTION B – everything on main** | *Rebase approach* (linear history):<br>`git rebase main`<br><br>*Merge approach* (shared branch):<br>`git merge --no-ff main` | *Rebase onto* main **or** *Merge* main |
| 7 | **Handle conflicts (if any)** | `git status` shows `UU` files → edit, `git add`, then `git rebase --continue` or finish merge | Red conflict markers; use built‑in 3‑way merge tool |
| 8 | **Restore stashed work** | `git stash pop` | *Git → Unstash* |
| 9 | **Push branch** | If rebased: `git push --force-with-lease`<br>If merged/cherry‑picked: `git push` | *Push* |
| 10 | **Update your Conda env locally** | `conda env update -f environment.yml --prune` | Terminal tab |

---

### 3. Choosing between cherry‑pick, rebase and merge

| Strategy | When to use | Pros | Cons |
|----------|-------------|------|------|
| **Cherry‑pick** just the env change | Only one isolated commit matters | Minimal history noise, super fast | Duplicate commit appears when `main` is merged later (harmless but visible) |
| **Rebase** branch onto `main` | You want a clean, linear history | No merge commits, easy bisects | Requires `git push --force-with-lease`; collaborators must sync |
| **Merge** `main` into branch | Branch is shared or history rewrite is risky | No force‑push, preserves exact history | Extra merge commit →

---

## 4. Tips & nice‑to‑haves

* `git config rebase.autoStash true` – Git will auto‑stash/unstash during rebase.
* Aliases you may like:  
  ```bash
  git config --global alias.up "pull --ff-only"
  git config --global alias.sync "!f() { git switch main && git up && git switch -; }; f"
  ```
* For CI: add a job that fails if `environment.yml` in a PR differs from main’s, reminding contributors to run this workflow.

---

## 5. FAQ

**Q: The cherry‑picked commit shows up twice after I merge the PR!**  
A: That’s expected—Git gave it a new hash on your branch.  
   When you _rebase_ later you can drop the duplicate in an interactive rebase
   (`git rebase -i main` → delete the line).

**Q: Somebody force‑pushed main and my pull failed.**  
A: Stop, talk to your team. Once history is rewritten you may need `git fetch --all` and a hard reset or rebase onto the new tip.

**Q: Conda still says packages are missing.**  
A: Run `conda clean -i` then repeat the `conda env update` with `--prune`.  
   And ensure you’re in the right environment (`conda activate <env>`).

---

*Last updated: 20 June 2025*

