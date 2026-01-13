#OPTUNA
python rl_files/run_optuna_parallel.py \
    --n-workers 2 \
    --train-script rl_files/train_bace_clean.py \
    --timeout-hours 72 \
    --study-name bace_baseline \
    --timesteps-per-trial 7_000_000 \
    --n-trials 300


## FINAL STATISTIC TESTING:

python rl_files/evaluate_single_model_multi_seed.py \
    --model-path runs_sb3/config0_v2/config0_v2_seed42_20260108_223909/best_model.zip \
    --eval-seeds 64 128 168 \
    --episodes 300 \
    --renderize 1


#Test Curriculum Training Run
python rl_files/train_bace_clean_curriculum.py \
  --seed 42 \
  --total-timesteps 400_000 \
  --experiment-name config0_v5 

#CONFIG 1 - Clean PPO Training
python rl_files/train_bace_clean.py \
  --seed 42 \
  --total-timesteps 10_000_000 \
  --experiment-name config0_v5 \
  --use_enriched_obs \
  --expert_alignment


#CORRECT PLOTTING
python rl_files/plot_eval_learning_curves.py \
    --experiment-dir runs_sb3/config0_v3 \
    --output-file config0_v3_LearningCurve.png \
    --title "Configuration 0: Baseline PPO" \
    --hyperparams "initial lr=2.5e-4, final lr=3.5e-5, initial ent=4.5e-2, final ent=2.5e-2, GAE lam=0.911, clip range=0.2, vf coef=0.77"

python rl_files/evaluate_best_model_300.py \
  --model-path runs_sb3/config0_baseline/config0_baseline_seed42_20260108_155218/best_model.zip \
  --renderize 1 \
  --episodes 300

python rl_files/evaluate_best_model_300.py \
  --model-path runs_sb3/config3_v1/config3_v1_seed42_20260107_103814/best_model.zip \
  --episodes 300

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