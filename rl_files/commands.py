#OPTUNA
python rl_files/run_optuna_parallel.py \
    --n-workers 2 \
    --train-script rl_files/train_bace_clean.py \
    --timeout-hours 72 \
    --study-name bace_baseline \
    --timesteps-per-trial 7_000_000 \
    --n-trials 300


## FINAL STATISTIC TESTING/EVALUATION:

python rl_files/evaluate_single_model_multi_seed.py \
    --model-path runs_sb3/baseline_experiment_v3/baseline_experiment_v3_seed42_20260115_122150/best_checkpoint/best_model.zip  \
    --vecnormalize-path runs_sb3/baseline_experiment_v3/baseline_experiment_v3_seed42_20260115_122150/best_checkpoint/best_vecnormalize.pkl \
    --eval-seeds 64 \
    --episodes 100 \
    --renderize 1


# EXPERIMENTAL RUN WITH SEEDS 42, 125, 456, 789, 1011
python rl_files/run_experiment.py \
    --experiment-name experiment_ABC \
    --total-timesteps 10_000_000 \
    --seeds 42 \
    --log-to-file


#CONFIG 1 - Clean PPO Training
python rl_files/train_bace_clean.py \
  --experiment-name baseline_experiment_v8 \
  --seed 123, 456, 789, 1011 \
  --total-timesteps 10000000
  --use-enriched-obs \
  --ablation-config all \
  --expert_alignment


#PLOTTING SECTION - Seed 125, 456, 789, 1011

python rl_files/plot_eval_learning_curve_updated.py \
    --experiment-dir runs_sb3/baseline_experiment_v4 \
    --output-file baseline_experiment_v4_LearningCurve_seed42.png \
    --title "Baseline Seed 42" \
    --hyperparams "initial lr=2.5e-4, final lr=3.5e-5, initial ent=4.5e-2, final ent=2.5e-2, GAE lam=0.911, clip range=0.2, vf coef=0.77"

# Plot only seed123
python rl_files/plot_eval_learning_curve_updated.py \
    --experiment-dir runs_sb3/baseline_experiment_v9 \
    --output-file baseline_v9_seed42.png \
    --seeds seed42 \
    --title "Test -High decay to low Entropy Coeff with closing reward improved"

# Plot multiple specific seeds
python rl_files/plot_eval_learning_curve_updated.py \
    --experiment-dir runs_sb3/baseline_experiment_v6/ \
    --output-file baseline_seeds_42_v6.png \
    --seeds seed42 \
    --title "Baseline Seeds 42 & 123"

# Plot all seeds (default behavior, unchanged)
python rl_files/plot_eval_learning_curve_updated.py \
    --experiment-dir runs_sb3/baseline_experiment_v3/ \
    --output-file baseline_all_seeds.png \
    --title "Baseline All Seeds"

#PLOT AVERAGE AMONG SSEEDS
python rl_files/plot_multiseed_learning_curve.py \
    --experiment-dir runs_sb3/baseline_experiment_v1 \
    --output-file baseline_mean_5seeds.png \
    --title "Baseline (5 Seeds)"





python rl_files/plot_eval_json.py \
    --json eval_300eps_20260102_104944.json \
    --outdir plots_eval




# With expert alignment (Configuration 2)
python rl_files/train_bace_config2.py --expert-alignment \
    --alignment-coef 0.001 \
    --alignment-decay-start 1500000 \
    --alignment-decay-end 6000000 \
    --total-timesteps 20_000_000 \
    --experiment-name config2_v2 \
    --seed 42


# Adjust alignment strength
python rl_files/train_bace_config2.py --expert-alignment --alignment-coef 0.15 --seed 42

# Adjust decay schedule (later start, longer decay)
python rl_files/train_bace_config2.py --expert-alignment \
    --alignment-decay-start 1000000 \
    --alignment-decay-end 3000000 \
    --seed 42



#CONFIG3 - Blending Schedule:
python rl_files/train_bace_config3.py --expert-blending \
    --initial-alpha 0.7 \
    --final-alpha 0.1 \
    --alpha-decay-start 1000000 \
    --alpha-decay-end 6000000 \
    --alpha-decay-type cosine \
    --experiment-name config3_v1 \
    --total-timesteps 10_000_000 \
    --seed 42



#Test Expert Standalone
python rl_files/test_config3_blending.py
python rl_files/test_expert_standalone.py --episodes 10 --render --verbose
python rl_files/test_expert_standalone.py --config configs/Scen_1_config.json --episodes 20