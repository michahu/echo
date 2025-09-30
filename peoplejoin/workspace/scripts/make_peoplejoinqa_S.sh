# train
for name in behavior_monitoring tracking_orders driving_school cinema school_bus; do
  python -m scripts.make_exp_order --n 10 --length max --config_dir workspace/peoplejoin-qa/experiments/exp_configs_test/$name
done

# test
for name in department_store election roller_coaster coffee_shop student_assessment; do
  python -m scripts.make_exp_order --n 10 --length max --config_dir workspace/peoplejoin-qa/experiments/exp_configs_test/$name
done