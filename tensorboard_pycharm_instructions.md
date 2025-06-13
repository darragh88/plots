# Launching TensorBoard from the PyCharm Terminal (Conda Environment)

Follow these steps inside **PyCharm’s built‑in Terminal** (bottom of the IDE window) to launch TensorBoard that uses the **same Conda environment** as your Jupyter notebook.

> Replace `sys-power-interns` with whatever environment name appears in `conda env list` if it’s different.

---

1. **Open the Terminal tab**  
   PyCharm starts it in the project’s root directory (where your `lightning_logs/` folder lives).

2. **List your Conda environments** *(just to verify the name)*  
   ```powershell
   conda env list
   ```
   Look for the environment created from `environment.yml`, e.g. **`sys-power-interns`**.

3. **Activate that environment**  
   ```powershell
   conda activate sys-power-interns
   ```

4. **(First time only) Install / upgrade TensorBoard inside the env**  
   ```powershell
   python -m pip install -U tensorboard
   ```
   *Skip this step on subsequent runs.*

5. **Launch TensorBoard, pointing it at Lightning’s default log directory**  
   ```powershell
   python -m tensorboard --logdir lightning_logs --port 6007
   ```
   - `python -m …` ensures you’re using the env’s Python even if `tensorboard` isn’t on the PATH.  
   - Change `--port` if 6007 is taken.

6. **Open the URL printed in the terminal**  
   PyCharm usually makes it clickable, e.g. <http://localhost:6007>.  
   Your browser will show training curves like **`train_loss`** and **`val_loss`**.

7. **Stop TensorBoard when you’re done**  
   Press **`Ctrl + C`** in the Terminal tab.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| *“python is not found”* | Open *Settings ▸ Advanced App Settings ▸ App Execution Aliases* and disable the Microsoft‑Store `python.exe` alias **or** always prefix commands with `conda run -n sys-power-interns …`. |
| *Port already in use* | Pick another port: `--port 6010` |