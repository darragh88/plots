# Quick‑Fix Guide: Getting TensorBoard Working on Windows (PyCharm + Jupyter)

Use **any one** of these methods—the first that works for you is enough.

---
## ❶ Zero‑Setup: Run TensorBoard *inside* the Jupyter notebook

```python
import sys, subprocess, importlib.util

# Install TensorBoard into **this** Jupyter kernel if missing
if importlib.util.find_spec("tensorboard") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "tensorboard"])

# Launch TensorBoard on an unused port (6007)
%load_ext tensorboard
%tensorboard --logdir lightning_logs --port 6007
```
*Why it works* – the `sys.executable -m pip …` line installs into the **exact interpreter** your notebook is running, so you never juggle Conda paths.

---
## ❷ PyCharm Terminal that auto‑activates your Conda env

1. **PyCharm ▸ Settings ▸ Tools ▸ Terminal**  
   ‑ *Shell path* →  
   ```
   "C:\Users\<you>\anaconda3\Scripts\activate.bat" sys-power-interns & powershell.exe
   ```

2. Open a **new** Terminal tab and run  
   ```powershell
   (sys-power-interns) PS > python -m tensorboard --logdir lightning_logs --port 6007
   ```

---
## ❸ No‑activation fallback: `conda run`

```powershell
conda run -n sys-power-interns python -m pip install -U tensorboard
conda run -n sys-power-interns python -m tensorboard --logdir lightning_logs --port 6007
```

Works even if `conda` isn’t on PATH or the env isn’t activated.

---
## ❹ Classic Anaconda Prompt

```powershell
conda activate sys-power-interns
python -m pip install -U tensorboard
python -m tensorboard --logdir lightning_logs --port 6007
```

---
## Diagnostics

```python
import sys, importlib.util
print("Python exe :", sys.executable)
print("TensorBoard installed?", importlib.util.find_spec("tensorboard") is not None)
```

---
## Common Windows Pitfalls & Fixes

| Symptom                          | Fix |
|----------------------------------|-----|
| `python` opens Microsoft Store   | Settings ▸ **App execution aliases** → disable *python.exe* |
| `conda` not found in PyCharm Terminal | Configure Terminal shell path as shown in **❷** or open *Anaconda Prompt* |
| Multiple Jupyter kernels         | Check `sys.executable`; install packages into that interpreter |

---
Once TensorBoard starts you’ll see something like:

```
TensorBoard 2.x at http://localhost:6007/ (Press CTRL+C to quit)
```

Open that URL to view **train_loss** and **val_loss** curves from your Lightning runs.