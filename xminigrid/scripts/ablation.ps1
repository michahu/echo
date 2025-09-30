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
for ($env_id = 6; $env_id -le 10; $env_id++) {

    # seed advances the iteration
    for ($seed = 1; $seed -le 16; $seed++) {
        $results_file = "results/$benchmark/$model/$agent/$env_id/results_$seed.json"
        $hindsight_file = "results/$benchmark/$model/$agent/$env_id/hindsight_${seed}.json"

        # echo env id and seed
        Write-Host "Running for env_id $env_id and seed $seed"

        # Skip first command if results file exists
        if (-not (Test-Path $results_file)) {
            python run.py --seed $seed --num-trials 1 --env-name $env_name --agent-type react --save-dir "results/$benchmark/$model/$agent/$env_id" --model $model --ruleset-id $env_id --hindsight-type $agent --max-steps 64 --obs-style text --llm-backend "default" --only-pickup-goals --save-movie --hindsight-k 16
        } else {
            Write-Host "Skipping run.py for seed $seed env_id $env_id - results file already exists: $results_file"
        }
    }
}

# ---------------------- manual control

# uv run python manual_control.py --env-id "XLand-MiniGrid"
