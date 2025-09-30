for task in department_store coffee_shop student_assessment wedding election; do
    for alg in echo; do
        bash scripts/eval.sh workspace/stateful-no-history/$alg/none/$task
    done
done