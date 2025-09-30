param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$agent,
    
    [Parameter(Mandatory=$true, Position=1)]
    [string]$model,
    
    [Parameter(Mandatory=$true, Position=2)]
    [string]$env_name
)

# powershell scripts/run.ps1 gce dev-gpt-4o-2024-05-13 XLand-MiniGrid-R4-17x17

# python run.py --seed 1 --num-trials 1 --env-name XLand-MiniGrid-R4-17x17 --agent-type react --save-dir "results/dev-gpt-4o-2024-05-13/gce/0" --model dev-gpt-4o-2024-05-13 --ruleset-id 0 --hindsight-type echo --max-steps 64 --obs-style text --llm-backend "default" --only-pickup-goals --save-movie

# for agent in react reflexion; do
for ($env_id = 12; $env_id -le 15; $env_id++) {

    # seed advances the iteration
    for ($seed = 0; $seed -le 15; $seed++) {
        $results_file = "results/$benchmark/$model/$agent/$env_id/results_$seed.json"
        $hindsight_file = "results/$benchmark/$model/$agent/$env_id/hindsight_${seed}.json"

        # Skip first command if results file exists
        if (-not (Test-Path $results_file)) {
            python run.py --seed $seed --num-trials 1 --env-name $env_name --agent-type react --save-dir "results/$benchmark/$model/$agent/$env_id" --model $model --ruleset-id $env_id --hindsight-type $agent --max-steps 64 --obs-style text --llm-backend "default" --only-pickup-goals --save-movie --hindsight-k 16 --env-id $env_id
        } else {
            Write-Host "Skipping run.py for seed $seed env_id $env_id - results file already exists: $results_file"
        }

        # Skip second command if hindsight file exists
        $retryCount = 0
        $maxRetries = 3
        while ($retryCount -lt $maxRetries -and -not (Test-Path $hindsight_file)) {
            try {
            python offline_compute.py --run-dir "results/$benchmark/$model/$agent/$env_id" --trial-file "results_${seed}.json" --mode $agent
            break
            }
            catch {
            $retryCount++
            Write-Host "Attempt $retryCount failed for offline_compute.py (seed $seed env_id $env_id): $($_.Exception.Message)"
            if ($retryCount -lt $maxRetries) {
                Write-Host "Retrying..."
                Start-Sleep -Seconds 2
            }
            }
        }
        if (Test-Path $hindsight_file) {
            Write-Host "Successfully completed offline_compute.py for seed $seed env_id $env_id"
        } elseif ($retryCount -eq $maxRetries) {
            Write-Host "Failed to complete offline_compute.py for seed $seed env_id $env_id after $maxRetries attempts"
        } else {
            Write-Host "Skipping offline_compute.py for seed $seed env_id $env_id - hindsight file already exists: $hindsight_file"
        }
    }
}

# ---------------------- manual control

# uv run python manual_control.py --env-id "XLand-MiniGrid"
