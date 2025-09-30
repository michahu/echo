# PowerShell version of run_peoplejoinqa_S.sh
# Usage: .\scripts\run_peoplejoinqa_S.ps1 37901 baseline full

param(
    [Parameter(Mandatory=$true)]
    [string]$PORT,
    
    [Parameter(Mandatory=$true)]
    [string]$HISTORY
)

# cre_Drama_Workshop_Groups department_store
foreach ($line_nums in 1, 2, 3) {
    # for alg in baseline reflexion; do
    foreach ($alg in @("baseline", "reflexion", "awm", "echo_entities", "one_traj_only", "echo")) {
        # for name in behavior_monitoring tracking_orders driving_school cinema school_bus; do
        # foreach ($name in @("department_store", "coffee_shop")) {
        foreach ($name in @("roller_coaster", "election", "student_assessment")) {
            Write-Host "Running: python scripts\run_experiments.py max $PORT $alg $name $HISTORY $line_nums"
            $env:PYTHONPATH = "src"
            & python "scripts\run_experiments.py" "max" $PORT $alg $name $HISTORY $line_nums
        }
    }
}
