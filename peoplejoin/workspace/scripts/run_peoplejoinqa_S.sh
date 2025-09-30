# bash scripts/run_peoplejoinqa_S.sh 37901 baseline full

PORT=$1
HISTORY=$2
ALG=$3


for line_nums in 1 2 3;  do
    for name in department_store coffee_shop student_assessment wedding election; do
        bash scripts/run_stateful_experiments.sh max $PORT $ALG $name $HISTORY $line_nums
    done
done

