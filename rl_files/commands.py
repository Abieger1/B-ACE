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

# Evaluate all seeds in an experiment:
python rl_files/evaluate_experiment_seeds.py \
--experiment-dir runs_sb3/experiment_D_2 \
--episodes 200

python rl_files/evaluate_experiment_seeds.py \
  --experiment-dir runs_sb3/experiment_D_2 \
  --episodes 200 \
  --speed-up 500




# EXPERIMENTAL RUN WITH SEEDS 42, 123, 456, 789, 1011
python rl_files/run_experiment.py \
    --experiment-name experiment_D_2 \
    --total-timesteps 10_000_000 \
    --seeds 123 456 789 \
    --expert-alignment \
    --alignment-coef 0.01

    
    --use-enriched-obs \
    --ablation-config 
    



#CONFIG 1 - Clean PPO Training
python rl_files/train_bace_clean.py \
  --experiment-name experiment_E \
  --seed 42 \
  --total-timesteps 10000000 \
  --expert-blending \
  --use-enriched-obs \
  --ablation-config all

  --expert-alignment \
  --alignment-coef 0.1
  
  --use-enriched-obs \
  --expert_alignment

python rl_files/train_bace_clean_expert_alignment.py \
  --experiment-name experiment_D \
  --seed 42 \
  --total-timesteps 10000000 \
  --expert-alignment

  --use-enriched-obs \


#Configurations:
# All enriched features (default when --use-enriched-obs is set)
python train_bace_clean_fixed.py --use-enriched-obs --seed 42

# Geometry only (disable all engagement features)
python train_bace_clean_fixed.py --use-enriched-obs \
    --disable-bez --disable-dmc --disable-offense-wez --disable-offense-ttc \
    --seed 42

# Engagement only (disable geometry)
python train_bace_clean_fixed.py --use-enriched-obs \
    --disable-apollonius \
    --seed 42

# DMC only
python train_bace_clean_fixed.py --use-enriched-obs \
    --disable-apollonius --disable-bez --disable-offense-wez --disable-offense-ttc \
    --seed 42

# With expert alignment + specific features
python train_bace_clean_fixed.py --use-enriched-obs \
    --disable-offense-ttc --disable-multi-threat \
    --expert-alignment --alignment-coef 0.1 \
    --seed 42

# Baseline (no enriched features)
python train_bace_clean_fixed.py --seed 42


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
    --experiment-dir runs_sb3/experiment_Baseline/ \
    --output-file baseline_seeds.png \
    --seeds seed42 seed123 seed456 seed1011 \
    --title "Baseline Seeds"

# Plot all seeds (default behavior, unchanged)
python rl_files/plot_eval_learning_curve_updated.py \
    --experiment-dir runs_sb3/experiment_ABC_V3/ \
    --output-file experiment_ABC_V3.png \
    --title "Experiment D - All Seeds"

#PLOT AVERAGE AMONG SSEEDS
python rl_files/plot_multiseed_learning_curve.py \
    --experiment-dir runs_sb3/baseline_experiment_v8 \
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
python rl_files/test_expert_standalone.py --expert hunter_fixed --episodes 20 --render --verbose

python rl_files/test_expert_standalone.py --expert all --episodes 20
# Focused sweep (10 configs) - good starting point
python rl_files/expert_sweep_bace.py --config b_ace_py/SimpleExample_B_ACE_config.json --episodes 20

# More episodes for better statistics
python expert_sweep_bace.py --config your_config.json --episodes 50

# Full grid search (256 configs - will take a while!)
python rl_files/expert_sweep_bace.py --config b_ace_py/SimpleExample_B_ACE_config.json --episodes 20 --full-sweep

# Verbose mode to see each episode
python expert_sweep_bace.py --config your_config.json --episodes 10 --verbose

python rl_files/test_expert_standalone.py --config b_ace_py/SimpleExample_B_ACE_config.json --episodes 20