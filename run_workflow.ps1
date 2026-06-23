# run_workflow.ps1
# ================
# Interactive launcher for the Aircraft Pitch RL project.
# Manages venv setup, tuning, training, evaluation and reporting.

trap {
    Write-Host "`n[ERROR] $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
$ErrorActionPreference = "Stop"

$PROJECT_DIR   = $PSScriptRoot
$CODE_DIR      = "$PROJECT_DIR\CODICE_PY"
$OUTPUT_DIR    = "$PROJECT_DIR\OUTPUT"
$VENV_ACTIVATE = "$PROJECT_DIR\venv_aircraft\Scripts\Activate.ps1"

# Episode constants - update if AircraftPitchEnv changes.
$EPISODE_STEPS   = 2400
$DT              = 0.05
$EPISODE_SECONDS = 120
$DQN_TUNE_STEPS  = 120000
$PPO_TUNE_STEPS  = 60000
$TUNE_TRIALS     = 40

# ------------------------------------------------------------------ #
#  Output helpers                                                       #
# ------------------------------------------------------------------ #

function Print-Header($msg) {
    Write-Host "`n$('='*60)" -ForegroundColor Cyan
    Write-Host "  $msg"      -ForegroundColor Cyan
    Write-Host "$('='*60)"   -ForegroundColor Cyan
}
function Print-OK($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green  }
function Print-Warn($msg) { Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function Print-Err($msg)  { Write-Host "  [XX] $msg" -ForegroundColor Red    }
function Print-Info($msg) { Write-Host "  [..] $msg" -ForegroundColor Gray   }

# ------------------------------------------------------------------ #
#  Virtual environment setup                                            #
# ------------------------------------------------------------------ #

$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Print-OK "Execution policy updated."
}

Set-Location $PROJECT_DIR
$env:PYTHONPATH = $CODE_DIR
$env:MPLBACKEND = "Agg"

if (-Not (Test-Path $VENV_ACTIVATE)) {
    Print-Warn "Virtual environment not found, creating now..."
    if (-Not (Get-Command python -ErrorAction SilentlyContinue)) {
        Print-Err "Python not found. Install from https://python.org"
        Read-Host "Press Enter to exit"; exit 1
    }
    python -m venv venv_aircraft
    if ($LASTEXITCODE -ne 0) { Print-Err "venv creation failed."; Read-Host "Press Enter to exit"; exit 1 }
    & $VENV_ACTIVATE
    python -m pip install --upgrade pip -q
    python -m pip install gymnasium stable-baselines3[extra] matplotlib numpy pillow optuna pytest scipy -q
    if ($LASTEXITCODE -ne 0) { Print-Err "Dependency installation failed."; exit 1 }
    Print-OK "Environment ready."
} else {
    & $VENV_ACTIVATE
}

# ------------------------------------------------------------------ #
#  Utility functions                                                    #
# ------------------------------------------------------------------ #

function Get-LatestModel($algo) {
    # Prefer the best_model checkpoint; fall back to the most recent .zip.
    $best = "$OUTPUT_DIR\models\${algo}_best\best_model.zip"
    if (Test-Path $best) {
        Print-OK "Using best model: ${algo}_best\best_model.zip"
        return ($best -replace "\.zip$", "")
    }
    $latest = Get-ChildItem "$OUTPUT_DIR\models\*${algo}*.zip" -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { Print-Err "No '$algo' model found. Train first."; return $null }
    Print-OK "Model: $($latest.Name)  [$($latest.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))]"
    return ($latest.FullName -replace "\.zip$", "")
}

function Get-LastValidJsonlRecord($path) {
    if (-Not (Test-Path $path)) { return $null }

    $lines = Get-Content $path -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($null -eq $lines) { return $null }
    if ($lines -isnot [System.Array]) { $lines = @($lines) }

    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        $clean = ($lines[$i] -replace "`0", "").Trim()
        if ([string]::IsNullOrWhiteSpace($clean)) { continue }
        try {
            return ($clean | ConvertFrom-Json)
        } catch {
            Print-Warn "Riga JSONL corrotta ignorata in $path (riga $($i+1)): $_"
            continue
        }
    }
    return $null
}

function Ask-Turbulence {
    # Prompt the user to choose a turbulence mode for training or evaluation.
    Write-Host "  Turbulence: [1] light  [2] moderate  [3] severe  [4] random  [5] curriculum (light->severe, gradual)" -ForegroundColor Cyan
    $t = (Read-Host "  Choice [1-5, default=2]").Trim()
    switch ($t) {
        "1" { return @{ Turbulence="light";    Curriculum=$false } }
        "2" { return @{ Turbulence="moderate"; Curriculum=$false } }
        "3" { return @{ Turbulence="severe";   Curriculum=$false } }
        "4" { return @{ Turbulence="random";   Curriculum=$false } }
        "5" { return @{ Turbulence="random";   Curriculum=$true  } }
        default { return @{ Turbulence="moderate"; Curriculum=$false } }
    }
}

# ------------------------------------------------------------------ #
#  Sanity check - 3 random episodes                                     #
# ------------------------------------------------------------------ #

function Run-SanityCheck {
    Print-Header "SANITY CHECK - 3 random episodes"

    $tmp = "$env:TEMP\sanity_check.py"
    $py = @()
    $py += "import numpy as np, sys"
    $py += "sys.path.insert(0, r'" + $CODE_DIR + "')"
    $py += "from aircraft_pitch_env import AircraftPitchEnv"
    $py += ""
    $py += "env = AircraftPitchEnv(turbulence_severity='moderate')"
    $py += "short_count = 0"
    $py += ""
    $py += "for ep in range(3):"
    $py += "    obs, _ = env.reset()"
    $py += "    total_r = 0.0"
    $py += "    n = 0"
    $py += "    done = False"
    $py += "    while not done:"
    $py += "        action = env.action_space.sample()"
    $py += "        obs, r, term, trunc, info = env.step(action)"
    $py += "        total_r += r"
    $py += "        n += 1"
    $py += "        done = term or trunc"
    $py += "    if n < 10:"
    $py += "        short_count += 1"
    $py += "    alt = info.get('altitude', 0)"
    $py += "    print(f'  ep{ep+1}: steps={n}  reward={total_r:.1f}  alt={alt:.0f}m')"
    $py += ""
    $py += "if short_count >= 2:"
    $py += "    print('SANITY_FAIL: env crashing immediately')"
    $py += "    sys.exit(1)"
    $py += "print('SANITY_OK')"
    $py | Set-Content $tmp -Encoding UTF8

    python $tmp
    if ($LASTEXITCODE -ne 0) {
        Print-Err "Sanity check failed - fix the environment before tuning/training."
        return $false
    }
    Print-OK "Sanity check passed."
    return $true
}

# ------------------------------------------------------------------ #
#  Training                                                             #
# ------------------------------------------------------------------ #

function Run-Train($algo, $turbulence, $curriculum) {
    $envs  = if ($algo -eq "ppo") { "24 envs" } else { "1 env" }
    $label = $algo.ToUpper()
    Print-Header "TRAINING $label - 1000000 steps - $envs"
    if ($curriculum) {
        Print-Info "Turbulence: curriculum - weights L/M/S shift gradually toward severe"
    } else {
        Print-Info "Turbulence: $turbulence (fixed for entire training)"
    }
    Print-Info "Episode: $EPISODE_STEPS steps = $EPISODE_SECONDS s"

    $a = @("$CODE_DIR\train.py", "--algo", $algo, "--timesteps", 1000000,
           "--turbulence", $turbulence, "--lr-schedule", "linear",
           "--output-dir", "$OUTPUT_DIR")
    if ($curriculum) { $a += "--curriculum" }
    python @a

    if ($LASTEXITCODE -eq 0) { Print-OK "$label training complete." }
    else                     { Print-Err "$label training failed (exit $LASTEXITCODE)." }
}

# ------------------------------------------------------------------ #
#  Evaluation                                                           #
# ------------------------------------------------------------------ #

function Run-Eval($algo, $turbulence = "moderate", $episodes = 20, $outDir = $null) {
    if ($null -eq $outDir) { $outDir = $OUTPUT_DIR }
    $label = $algo.ToUpper()
    Print-Header "EVAL $label - $episodes eps - turbulence: $turbulence"
    $model = Get-LatestModel $algo
    if ($null -eq $model) { return }

    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $evalDir = "$outDir\plots\${algo}_${turbulence}_${timestamp}"
    python "$CODE_DIR\evaluate.py" `
        --model "$model" --algo $algo `
        --episodes $episodes `
        --turbulence $turbulence `
        --output-dir "$outDir" `
        --plot-dir "$evalDir"

    if ($LASTEXITCODE -eq 0) { Print-OK "Done. Plots in $outDir\plots\" }
    else                     { Print-Err "Evaluation failed (exit $LASTEXITCODE)." }
}

function Run-LongTest {
    <#
    .SYNOPSIS
        Episodio di test da 2h con severita' turbolenza (light/moderate/severe)
        che cambia casualmente a ogni segmento.
 
    .PARAMETER Algo
        "dqn" o "ppo" (default: chiede all'utente).
 
    .PARAMETER Seed
        Seed per la sequenza di segmenti (default: 42).
 
    .PARAMETER SegMin
        Durata minima di ogni segmento in minuti (default: 8).
 
    .PARAMETER SegMax
        Durata massima di ogni segmento in minuti (default: 18).
 
    .DESCRIPTION
        Invoca long_test_episode.py con il modello migliore disponibile.
        Salva plot e JSONL in OUTPUT/plots/ e OUTPUT/logs/.
    #>
    param(
        [string]$Algo   = "",
        [int]   $Seed   = 42,
        [float] $SegMin = 8.0,
        [float] $SegMax = 18.0
    )
 
    # --- scegli algo se non passato ---
    if ($Algo -eq "") {
        $choice = (Read-Host "  Alg: [1] DQN  [2] PPO  (default=1)").Trim()
        $Algo   = if ($choice -eq "2") { "ppo" } else { "dqn" }
    }
 
    # --- parametri interattivi se invocato senza argomenti ---
    $seedIn   = (Read-Host "  Seed severity sequence [default=$Seed]").Trim()
    if ($seedIn -ne "") { $Seed = [int]$seedIn }
 
    $segMinIn = (Read-Host "  min duration [default=$SegMin]").Trim()
    if ($segMinIn -ne "") { $SegMin = [float]$segMinIn }
 
    $segMaxIn = (Read-Host "  max duration [default=$SegMax]").Trim()
    if ($segMaxIn -ne "") { $SegMax = [float]$segMaxIn }
 
    # --- trova il modello ---
    $model = Get-LatestModel $Algo
    if ($null -eq $model) { return }
 
    Print-Header "LONG TEST 2H  -  $($Algo.ToUpper())  seed=$Seed  seg=$SegMin-$SegMax min"
    Print-Info "episode duration: 120 min = 144 000 step"
    Print-Info "Model           : $model"
    Print-Info "Output          : $OUTPUT_DIR\plots\  e  $OUTPUT_DIR\logs\long_test_runs.jsonl"
 
    python "$CODE_DIR\long_test_episode.py" `
        --model  "$model"       `
        --algo   $Algo          `
        --seed   $Seed          `
        --seg-min $SegMin       `
        --seg-max $SegMax       `
        --output-dir "$OUTPUT_DIR"
 
    if ($LASTEXITCODE -eq 0) {
        Print-OK "Long test completed."
        Print-OK "Plot  -> $OUTPUT_DIR\plots\"
        Print-OK "Log   -> $OUTPUT_DIR\logs\long_test_runs.jsonl"
    } else {
        Print-Err "Long test failed (exit $LASTEXITCODE)."
    }
}
 

# ------------------------------------------------------------------ #
#  Compare + Turbulence Matrix (unified)                                #
# ------------------------------------------------------------------ #

function Run-CompareMatrix {
    <#
    .SYNOPSIS
        Evaluates DQN and PPO across light / moderate / severe turbulence,
        prints a side-by-side comparison table, and saves a summary JSON.

    .DESCRIPTION
        For each (algo, turbulence) cell:
          - runs 30 deterministic episodes
          - saves per-episode plots under OUTPUT\compare_<algo>\
          - collects mean reward and std from the JSONL log

        After all cells are done a formatted table is printed to the console
        and results are written to OUTPUT\turbulence_matrix\summary.json.
    #>
    param([string]$Episodes = "30")

    $nEps = [int]$Episodes

    Print-Header "COMPARE DQN vs PPO  |  TURBULENCE MATRIX  ($($nEps) eps / cell)"

    $dqn = Get-LatestModel "dqn"
    $ppo = Get-LatestModel "ppo"

    if ($null -eq $dqn -and $null -eq $ppo) {
        Print-Err "No models found. Train at least one agent first."
        return
    }

    $levels  = @("light", "moderate", "severe")
    $algos   = @("dqn", "ppo")
    $results = @()

    foreach ($turb in $levels) {
        foreach ($algo in $algos) {
            $model = if ($algo -eq "dqn") { $dqn } else { $ppo }
            if ($null -eq $model) {
                Print-Warn "Skipping $algo / $turb  (no model found)"
                continue
            }

            $outDir = "$OUTPUT_DIR\compare_${algo}"
            Print-Info "Evaluating $($algo.ToUpper()) / $turb ..."

            python "$CODE_DIR\evaluate.py" `
                --model "$model" --algo $algo `
                --episodes $nEps `
                --turbulence $turb `
                --output-dir "$outDir"

            if ($LASTEXITCODE -ne 0) {
                Print-Warn "$algo / $turb evaluation failed (exit $LASTEXITCODE) - skipping cell."
                continue
            }

            # Read the last VALID JSONL record written by evaluate.py.
            $jsonlPath = "$outDir\logs\${algo}_runs.jsonl"
            $last = Get-LastValidJsonlRecord $jsonlPath
            if ($null -ne $last) {
                $results += [PSCustomObject]@{
                    algo        = $algo.ToUpper()
                    turbulence  = $turb
                    mean_reward = [math]::Round($last.mean_reward, 1)
                    std         = [math]::Round($last.std_reward,  1)
                    episodes    = $nEps
                }
            } else {
                Print-Warn "No valid JSONL record found for $algo / $turb  (path: $jsonlPath)"
            }
        }
    }

    # ---- print table -------------------------------------------------------
    if ($results.Count -gt 0) {
        Write-Host ""
        Write-Host "$('='*60)" -ForegroundColor Cyan
        Write-Host "  RESULTS  ($($nEps) episodes per cell)" -ForegroundColor Cyan
        Write-Host "$('='*60)" -ForegroundColor Cyan
        Write-Host ("  {0,-6}  {1,-10}  {2,-12}  {3}" -f "ALGO", "TURB", "MEAN REWARD", "+/- STD") `
            -ForegroundColor White

        foreach ($r in $results) {
            $col = switch ($r.turbulence) {
                "light"    { "Green"  }
                "moderate" { "Yellow" }
                "severe"   { "Red"    }
                default    { "Gray"   }
            }
            Write-Host ("  {0,-6}  {1,-10}  {2,-12}  +/- {3}" -f `
                $r.algo, $r.turbulence, $r.mean_reward, $r.std) -ForegroundColor $col
        }
        Write-Host "$('='*60)" -ForegroundColor Cyan
        Write-Host ""

        # ---- save JSON summary ---------------------------------------------
        $matrixDir = "$OUTPUT_DIR\turbulence_matrix"
        New-Item -ItemType Directory -Force -Path $matrixDir | Out-Null
        $results | ConvertTo-Json | `
            Set-Content "$matrixDir\summary.json" -Encoding UTF8
        Print-OK "Summary JSON -> OUTPUT\turbulence_matrix\summary.json"
    } else {
        Print-Warn "No results collected - nothing to display."
    }

    Print-OK "DQN plots -> OUTPUT\compare_dqn\plots\"
    Print-OK "PPO plots -> OUTPUT\compare_ppo\plots\"
}

# ------------------------------------------------------------------ #
#  Hyperparameter tuning                                                #
# ------------------------------------------------------------------ #

function Run-Tune($algo) {
    $steps = if ($algo -eq "dqn") { $DQN_TUNE_STEPS } else { $PPO_TUNE_STEPS }
    $eps   = if ($algo -eq "dqn") { [math]::Floor($steps / $EPISODE_STEPS) } `
             else                  { [math]::Floor($steps / $EPISODE_STEPS * 8) }
    $label = $algo.ToUpper()
    $sStr  = "{0:N0}" -f $steps
    $tStr  = "{0:N0}" -f ($TUNE_TRIALS * $steps)

    Print-Header "TUNING $label - $TUNE_TRIALS trials x $sStr steps = $tStr total"
    Print-Info "$eps complete episodes per trial"
    Print-Info "episode = $EPISODE_STEPS steps = $EPISODE_SECONDS s"

    python "$CODE_DIR\tune.py" --algo $algo --trials $TUNE_TRIALS `
        --timesteps $steps --output-dir "$OUTPUT_DIR"

    if ($LASTEXITCODE -eq 0) { Print-OK "Done. OUTPUT\reports\best_params_$algo.json" }
    else                     { Print-Err "Tuning failed (exit $LASTEXITCODE)." }
}

# ------------------------------------------------------------------ #
#  TensorBoard                                                          #
# ------------------------------------------------------------------ #

function Run-TensorBoard {
    $logs = "$OUTPUT_DIR\logs"
    if (-Not (Test-Path $logs)) { Print-Err "No logs found. Train first."; return }
    Print-OK "TensorBoard -> http://localhost:6006"
    Start-Process powershell -ArgumentList "-NoExit","-Command","tensorboard --logdir `"$logs`""
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:6006"
}

# ------------------------------------------------------------------ #
#  Project status                                                       #
# ------------------------------------------------------------------ #

function Show-ProjectStatus {
    Print-Header "PROJECT STATUS"
    foreach ($algo in @("dqn","ppo")) {
        $best       = "$OUTPUT_DIR\models\${algo}_best\best_model.zip"
        $p          = "$OUTPUT_DIR\reports\best_params_$algo.json"
        $log        = "$OUTPUT_DIR\logs\${algo}_runs.jsonl"
        $has_model  = if (Test-Path $best) { "best_model OK" } else { "no best model" }
        $has_params = if (Test-Path $p)    { "tuned"         } else { "defaults"      }
        $has_runs   = if (Test-Path $log)  {
            $last = (Get-Content $log -Encoding UTF8 | Select-Object -Last 1) | ConvertFrom-Json
            "last reward=" + $last.mean_reward + " +/- " + $last.std_reward
        } else { "no runs" }
        $col = if ($has_model -eq "best_model OK") { "Green" } else { "Yellow" }
        Write-Host ("  {0,-4}  {1}   params: {2}   {3}" -f $algo.ToUpper(), $has_model, $has_params, $has_runs) -ForegroundColor $col
    }
    Write-Host ""
    $plots = (Get-ChildItem "$OUTPUT_DIR\plots\*.png" -ErrorAction SilentlyContinue).Count
    Print-Info "$plots PNG files in OUTPUT\plots\"
    Print-Info "Episode: $EPISODE_STEPS steps x $DT s = $EPISODE_SECONDS s"
    Print-Info "Tune budget: DQN=$("{0:N0}" -f $DQN_TUNE_STEPS) steps/trial   PPO=$("{0:N0}" -f $PPO_TUNE_STEPS) steps/trial"
}


# ------------------------------------------------------------------ #
#  Autopilot [0] - full pipeline (tune + train + eval + matrix)         #
# ------------------------------------------------------------------ #

function Run-AutoPilot {
    Print-Header "AUTOPILOT [0] - unit tests + tune + train + compare matrix"

    Print-Info "Step 1/7: unit tests"
    python -m pytest "$CODE_DIR\test_env.py" -q --tb=line
    if ($LASTEXITCODE -ne 0) { Print-Warn "Some tests failed - continuing anyway." }

    Print-Info "Step 2/7: sanity check"
    $ok = Run-SanityCheck
    if (-not $ok) { Print-Err "Aborting autopilot: environment is not stable."; return }

    Print-Info "Step 3/7: tuning DQN"
    Run-Tune "dqn"

    Print-Info "Step 4/7: tuning PPO"
    Run-Tune "ppo"

    Print-Info "Step 5/7: training DQN"
    Run-Train "dqn" "random" $true

    Print-Info "Step 6/7: training PPO"
    Run-Train "ppo" "random" $true

    Print-Info "Step 7/7: compare + turbulence matrix (30 eps/cell)"
    Run-CompareMatrix -Episodes 30

    Print-Header "AUTOPILOT [0] COMPLETE"
    Print-OK "Models      -> OUTPUT\models\"
    Print-OK "Plots       -> OUTPUT\compare_dqn\  and  OUTPUT\compare_ppo\"
    Print-OK "Matrix      -> OUTPUT\turbulence_matrix\summary.json"
    Print-OK "Logs        -> OUTPUT\logs\"
    Print-OK "Reports     -> OUTPUT\reports\"
}

# ------------------------------------------------------------------ #
#  Autopilot [0b] - skip tuning (use existing or default params)        #
# ------------------------------------------------------------------ #

function Run-AutoPilotFast {
    Print-Header "AUTOPILOT [0b] - unit tests + train + compare matrix  (no tuning)"

    Print-Info "Step 1/5: unit tests"
    python -m pytest "$CODE_DIR\test_env.py" -q --tb=line
    if ($LASTEXITCODE -ne 0) { Print-Warn "Some tests failed - continuing anyway." }

    Print-Info "Step 2/5: sanity check"
    $ok = Run-SanityCheck
    if (-not $ok) { Print-Err "Aborting autopilot: environment is not stable."; return }

    Print-Info "Step 3/5: training DQN  (using existing tuned params or defaults)"
    Run-Train "dqn" "random" $true

    Print-Info "Step 4/5: training PPO  (using existing tuned params or defaults)"
    Run-Train "ppo" "random" $true

    Print-Info "Step 5/5: compare + turbulence matrix (25 eps/cell)"
    Run-CompareMatrix -Episodes 30

    Print-Header "AUTOPILOT [0b] COMPLETE"
    Print-OK "Models      -> OUTPUT\models\"
    Print-OK "Plots       -> OUTPUT\compare_dqn\  and  OUTPUT\compare_ppo\"
    Print-OK "Matrix      -> OUTPUT\turbulence_matrix\summary.json"
    Print-OK "Logs        -> OUTPUT\logs\"
    Print-OK "Reports     -> OUTPUT\reports\"
}

# ------------------------------------------------------------------ #
#  Main menu                                                            #
# ------------------------------------------------------------------ #

function Show-Menu {
    Print-Header "AIRCRAFT PITCH RL"
    Write-Host "  [0]   AUTOPILOT      (tests + tune DQN+PPO + train + compare matrix)" -ForegroundColor Magenta
    Write-Host "  [0b]  AUTOPILOT FAST (no tuning)"                                     -ForegroundColor Magenta
    Write-Host ""

    Write-Host "  TRAIN" -ForegroundColor Yellow
    Write-Host "    [1] DQN  (asks turbulence)"
    Write-Host "    [2] PPO  (asks turbulence)"
    Write-Host ""

    Write-Host "  TUNE & TEST" -ForegroundColor Yellow
    Write-Host "    [6] Tune DQN"
    Write-Host "    [7] Tune PPO"
    Write-Host "    [8] Unit tests"
    Write-Host "    [s] Sanity check"
    Write-Host ""

    Write-Host "  EVAL" -ForegroundColor Yellow
    Write-Host "    [3]  Eval DQN"
    Write-Host "    [4]  Eval PPO"
    Write-Host "    [5]  Compare DQN vs PPO + turbulence matrix"
    Write-Host "    [11] Long test 2h (random turbulence segments)"
    Write-Host ""

    Write-Host "  UTILITY" -ForegroundColor Yellow
    Write-Host "    [9]  TensorBoard"
    Write-Host "    [10] Project status"
    Write-Host ""

    Write-Host "    [q]  Exit"
    Write-Host ""
}

do {
    Show-Menu
    $choice = (Read-Host "  Choice").Trim()
    switch ($choice) {
        "0"   { Run-AutoPilot }
        "0b"  { Run-AutoPilotFast }

        "1"   { $c = Ask-Turbulence; Run-Train "dqn" $c.Turbulence $c.Curriculum }
        "2"   { $c = Ask-Turbulence; Run-Train "ppo" $c.Turbulence $c.Curriculum }

        "6"   { Run-Tune "dqn" }
        "7"   { Run-Tune "ppo" }
        "8"   { python -m pytest "$CODE_DIR\test_env.py" -v --tb=short; if ($LASTEXITCODE -eq 0) { Print-OK "All passed." } else { Print-Warn "Some failed." } }
        "s"   { Run-SanityCheck }

        "3"   { $c = Ask-Turbulence; Run-Eval "dqn" $c.Turbulence }
        "4"   { $c = Ask-Turbulence; Run-Eval "ppo" $c.Turbulence }
        "5"   { Run-CompareMatrix }
        "11"  { Run-LongTest }

        "9"   { Run-TensorBoard }
        "10"  { Show-ProjectStatus }

        "q"   { Print-OK "Exiting."; break }
        default { Print-Warn "Invalid choice '$choice'." }
    }
} while ($choice -ne "q")