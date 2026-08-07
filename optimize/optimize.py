from firecrest_executor import FirecrestExecutor
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from nettest import execute
import nevergrad as ng
import yaml
import math
import argparse
import json
import os
from pathlib import Path
from nettest.utils import MyDumper
from nettest.default_environment import get_default_environment

STATE_FILE = "optimization_state.json"
RANDOM_SEED = 42

DEFAULT_STATE = {
    "completed": [],
    "running": [],
    "nElo_target": 0.5,
    "next_exec_id": 1,
}


def load_state():
    state = DEFAULT_STATE.copy()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state.update(json.load(f))
    return state


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


class RemoteNet:
    def __init__(self, environment, max_workers=16, local=False):
        # we could probably share the executor between calls to train_and_test_net
        if local:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = FirecrestExecutor(
                working_dir="/users/vjoost/fish/workspace/",
                sbatch_options=[
                    "--job-name=FirecrestExecutor",
                    "--time=12:00:00",
                    "--nodes=1",
                    "--partition=normal",
                ],
                srun_options=[
                    "--environment=/users/vjoost/fish/workspace/nettest.toml"
                ],
                sleep_interval=10,
                max_workers=max_workers,
                task_retries=30,
            )
        self.environment = environment

    # the recipe to optimize
    def train_and_test_net(
        self,
        local_exec_id,
        nElo_target,
        both_lambda,
        pow_exp,
        qp_asymmetry,
        in_scaling,
        out_scaling,
        in_offset,
        out_offset_shift,
    ):
        print(
            f"Starting {local_exec_id}:",
            both_lambda,
            pow_exp,
            qp_asymmetry,
            in_scaling,
            out_scaling,
            in_offset,
            out_offset_shift,
            flush=True,
        )

        out_offset = in_offset + out_offset_shift

        recipe_str = f"""
training_binpacks: &training_binpacks
  - vondele/from_kaggle_2_relabel/T60T70wIsRightFarseerT60T74T75T76.split_0.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_2_relabel/T60T70wIsRightFarseerT60T74T75T76.split_1.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_2_relabel/T60T70wIsRightFarseerT60T74T75T76.split_2.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_2_relabel/T60T70wIsRightFarseerT60T74T75T76.split_3.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_2_relabel/T60T70wIsRightFarseerT60T74T75T76.split_4.relabel-BT4-tf13tune.binpack
  - vondele/master-binpacks_relabel/nodes5000pv2_UHO.relabel-BT4-tf13tune.binpack
  - vondele/master-binpacks_relabel/wrongIsRight_nodes5000pv2.relabel-BT4-tf13tune.binpack
  - vondele/master-binpacks_relabel/multinet_pv-2_diff-100_nodes-5000.relabel-BT4-tf13tune.binpack
  - vondele/master-binpacks_relabel/dfrc_n5000.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_1_relabel/leela96-filt-v2.min.split_0.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_1_relabel/leela96-filt-v2.min.split_1.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_1_relabel/leela96-filt-v2.min.split_2.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_1_relabel/leela96-filt-v2.min.split_3.relabel-BT4-tf13tune.binpack
  - vondele/from_kaggle_1_relabel/leela96-filt-v2.min.split_4.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test60-2021-11-nov-12tb7p.min-v2.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test60-2021-12-dec-12tb7p.min-v2.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test77-2021-12-dec-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test79-2022-04-apr-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test79-2022-05-may-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test80-2022-06-jun-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test80-2022-07-jul-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test80-2022-08-aug-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test80-2022-09-sep-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test80-2022-10-oct-16tb7p.v6-dd.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_1/test80-2022-11-nov-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-01-jan-16tb7p.v6-sk20.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-02-feb-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-03-mar-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-04-apr-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-05-may-2tb7p.v6.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-06-jun-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-07-jul-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-08-aug-2tb7p.v6.min.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-09-sep-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-10-oct-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-11-nov-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
  - vondele/linrock_relabel_2/test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
  - xushawn/test80-bt4-relabel/test80-2024-01-jan-2tb7p.min-v2.v6.relabel.binpack
  - xushawn/test80-bt4-relabel/test80-2024-02-feb-2tb7p.min-v2.v6.relabel.binpack
  - vondele/rescored/tb5dtm.binpack

model_config: &model_config
  features: Full_Threats+PP_3Wide+HalfKAv2_hm^
  l1: 1024
  l2: 32

optimize_flags: &optimize_flags
  <<: *model_config
  ft-optimize-count: 100000
  ft-optimize: true
  ft-compression: leb128

checkpoint2nnue_options: &checkpoint2nnue_options
  <<: *model_config
  ft-compression: leb128

common_run_options: &common_run_options
  optimizer-name: rangerlite
  epoch-size: 134217728
  validation-size: 268435456
  check-val-every-n-epoch: 50
  batch-size: 131072
  <<: *model_config
  #factorized-weight-decay: 0.001
  random-fen-skipping: 10
  early-fen-skipping: 18
  soft-early-fen_skipping: 32
  pc-y0: -0.20
  pc-y1: 0.45
  pc-y2: 1.0
  pc-y3: 0.95
  pc-y4: 0.75

advanced_stage_options: &advanced_stage_options
  ply-x1: 0.00
  ply-y1: 0.025
  ply-x2: 22.0
  ply-y2: 0.05
  ply-x3: 25.5
  ply-y3: 0.20
  ply-x4: 29.5
  ply-y4: 0.80
  pow-exp: 2.4340402395048404
  qp-asymmetry: 0.22866886710086187
  in-scaling: 295.6539508488627
  out-scaling: 379.98724077106635
  in-offset: 285.2706341467852
  out-offset: 289.1258344152218
  one-cycle-warmup-pct: 0.05
  one-cycle-final-div: "1e3"
  lambda-cycle-jitter: true
  jitter-lambda-sample: 0.0034284671121657256
  jitter-lambda-batch: 0.0061980291754002
  jitter-decay-lambda-batch: 0.999
  start-lambda: 1.0
  end-lambda: 1.0
  lambda-cycle-warmup-pct: 0.2
  lambda-cycle-delta: -0.3

trainer: &trainer
  owner: official-stockfish
  sha: 9f72946529c4187d3679014036cd22c3be419716

reference_code: &reference_code
  owner: official-stockfish
  sha: c5aef2bf1f77d94a3dd476f276af68fd71a0ac07
  target: profile-build

testing_code: &testing_code
  owner: official-stockfish
  sha: c5aef2bf1f77d94a3dd476f276af68fd71a0ac07
  target: profile-build

# --- Grouped Structure Anchors for Redundancy Reduction ---
common_convert: &common_convert
  binpack: official-stockfish/master-binpacks/fishpack32.binpack
  checkpoint2nnue: *checkpoint2nnue_options
  optimize: *optimize_flags

common_step: &common_step
  convert: *common_convert
  trainer: *trainer

common_run_resume_model: &common_run_resume_model
  repetitions: 1
  resume: previous_model

common_run_resume_ckpt: &common_run_resume_ckpt
  <<: *common_run_resume_model
  resume: previous_checkpoint

# ----------------------------------------------------------

testing:
  crosscheck:
    trainer: *trainer
    binpack: official-stockfish/master-binpacks/fishpack32.binpack
    other_options:
      <<: *model_config
      count: 20000
  fastchess:
    code:
      owner: Disservin
      sha: 74deac23aadc6718fa0f8e24ac95d16392d79444
    options:
      hash: 16
      max_rounds: 40000
      tc: 10+0.1
    sprt:
      nElo_interval_midpoint: {nElo_target}
      nElo_interval_width: 2

  reference:
    code: *reference_code
  steps: last
  testing:
    code: *testing_code


training:
  steps:
    # Pretraining
    - <<: *common_step
      run:
        resume: none
        binpacks: *training_binpacks
        max_epochs: 3500
        repetitions: 3
        other_options:
          <<: [*common_run_options, *advanced_stage_options]
          one-cycle-steps: 4096000
          lambda-schedule-steps: 3584000
          lr: "0.9e-3"

    # Finetuning
    - <<: *common_step
      trainer: *trainer
      run:
        <<: *common_run_resume_ckpt
        binpacks: *training_binpacks
        max_epochs: 3750
        other_options:
          <<: [*common_run_options, *advanced_stage_options]
          one-cycle-steps: 4096000
          lr: "0.9e-3"
          batch_size: 65536
          start-lambda: {both_lambda}
          end-lambda: {both_lambda}
          pow-exp: {pow_exp}
          qp-asymmetry: {qp_asymmetry}
          in-scaling: {in_scaling}
          out-scaling: {out_scaling}
          in-offset: {in_offset}
          out-offset: {out_offset}
          jitter-lambda-sample: 0.0
          jitter-lambda-batch: 0.0
          jitter-decay-lambda-batch: 0.0
        """
        recipe = yaml.safe_load(recipe_str)

        with Path(f"optimize_recipe_str_{local_exec_id}.yaml").open(
            mode="w", encoding="utf-8"
        ) as f:
            f.write(recipe_str)

        with Path(f"optimize_recipe_{local_exec_id}.yaml").open(
            mode="w", encoding="utf-8"
        ) as f:
            yaml.dump(recipe, f, Dumper=MyDumper, default_flow_style=False, width=300)

        bestNet, nElo = execute(
            recipe=recipe,
            executor=self.executor,
            environment=self.environment,
        )

        if nElo is None:
            # TODO ... better error handling possible ?
            print(
                "Something went wrong during evaluation .... continuing with lower bound estimate"
            )
            nElo = nElo_target - 10

        print(
            f"Done {local_exec_id}:",
            both_lambda,
            pow_exp,
            qp_asymmetry,
            in_scaling,
            out_scaling,
            in_offset,
            out_offset_shift,
            nElo,
            bestNet,
            flush=True,
        )

        return -nElo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize a recipe")
    parser.add_argument(
        "--environment", required=False, help="Definition of the environment file"
    )
    parser.add_argument(
        "--restart", action="store_true", help="Restart optimization from saved state"
    )
    args = parser.parse_args()

    if args.environment:
        print("Using environment file: ", args.environment)
        with open(args.environment) as f:
            environment = yaml.safe_load(f)
    else:
        environment = get_default_environment()

    instrumentation = ng.p.Instrumentation(
        # ng.p.Scalar(init=28)
        # .set_bounds(lower=10, upper=32)
        # .set_mutation(sigma=3.0)
        # .set_integer_casting(),  # early_fen_skipping
        # ng.p.Scalar(init=1.0)
        # .set_bounds(lower=0.5, upper=2.0)
        # .set_mutation(sigma=0.2),  # pc_y1
        # ng.p.Scalar(init=2.0)
        # .set_bounds(lower=1.0, upper=4.0)
        # .set_mutation(sigma=0.5),  # pc_y2
        # ng.p.Scalar(init=1.0)
        # .set_bounds(lower=0.5, upper=2.0)
        # .set_mutation(sigma=0.2),  # pc_y3
        # ng.p.Scalar(init=0.0)
        # .set_bounds(lower=-4.0, upper=4.0)
        # .set_mutation(sigma=1.0),  # lr_scaling_power
        # ng.p.Scalar(init=5.0)
        # .set_bounds(lower=0.0, upper=10.0)
        # .set_mutation(sigma=1.0),  # gamma_adjust
        # ng.p.Scalar(init=0.0030)  # lambda_sample
        # .set_bounds(lower=0.0000, upper=0.0090)
        # .set_mutation(sigma=0.002),
        # ng.p.Scalar(init=0.0100)  # lambda_batch
        # .set_bounds(lower=0.0000, upper=0.0300)
        # .set_mutation(sigma=0.006),
        # ng.p.Scalar(init=0.76)  # both_lambda
        # .set_bounds(lower=0.6, upper=0.9)
        # .set_mutation(sigma=0.02),
        # ng.p.Scalar(init=0.75)  # end_lambda
        # .set_bounds(lower=0.55, upper=0.85)
        # .set_mutation(sigma=0.04),
        #
        ng.p.Scalar(init=0.9)  # both_lambda
        .set_bounds(lower=0.6, upper=1.0)
        .set_mutation(sigma=0.1),
        ng.p.Scalar(init=2.442037790427722)  # pow_exp
        .set_bounds(lower=2.0, upper=3.0)
        .set_mutation(sigma=0.1),
        ng.p.Scalar(init=0.228)  # qp_asymmetry
        .set_bounds(lower=0.1, upper=0.3)
        .set_mutation(sigma=0.03),
        ng.p.Scalar(init=294.7)  # in_scaling
        .set_bounds(lower=214, upper=374)
        .set_mutation(sigma=20),
        ng.p.Scalar(init=352.8)  # out_scaling
        .set_bounds(lower=284, upper=444)
        .set_mutation(sigma=20),
        ng.p.Scalar(init=281.4)  # in_offset
        .set_bounds(lower=210, upper=330)
        .set_mutation(sigma=20),
        ng.p.Scalar(init=-2)  # out_offset_shift
        .set_bounds(lower=-30, upper=30)
        .set_mutation(sigma=10),
    )

    budget = 256  # Total number of evaluations to perform
    num_workers = 8  # Number of parallel workers to use

    instrumentation.random_state.seed(RANDOM_SEED)

    state = load_state()
    pending_candidates = []
    current_nElo_target = state["nElo_target"]
    next_exec_id = state["next_exec_id"]

    # The remotely trainable net
    remoteNet = RemoteNet(
        environment=environment,
        max_workers=num_workers,
        local=False,
    )

    # Use TBPSA optimizer
    optimizer = ng.optimizers.TBPSA(
        instrumentation, budget=budget, num_workers=num_workers
    )

    if args.restart:
        print(
            f"Restarting from state: {len(state['completed'])} completed, {len(state['running'])} running."
        )

        # Reload completed points into the optimizer
        for item in state["completed"]:
            cand = optimizer.parametrization.spawn_child()
            cand.value = (tuple(item["args"]), item["kwargs"])
            optimizer.tell(cand, item["loss"])

        # Queue previously running points for re-evaluation
        for item in state["running"]:
            cand = optimizer.parametrization.spawn_child()
            cand.value = (tuple(item["args"]), item["kwargs"])
            pending_candidates.append(
                (cand, item["exec_id"], item["nElo_target"])
            )
    else:
        save_state(state)

    active_futures = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while len(state["completed"]) < budget:
            # Fill the thread pool up to num_workers or remaining budget
            while (
                len(active_futures) < num_workers
                and (len(state["completed"]) + len(active_futures)) < budget
            ):
                if pending_candidates:
                    candidate, exec_id, point_target = pending_candidates.pop(0)
                    is_restart_point = True
                else:
                    candidate = optimizer.ask()
                    exec_id = next_exec_id
                    next_exec_id += 1
                    point_target = current_nElo_target
                    is_restart_point = False

                future = executor.submit(
                    remoteNet.train_and_test_net,
                    exec_id,
                    point_target,
                    *candidate.args,
                    **candidate.kwargs,
                )
                active_futures[future] = (candidate, exec_id)

                # Track and save the running point
                if not is_restart_point:
                    state["running"].append(
                        {
                            "args": candidate.args,
                            "kwargs": candidate.kwargs,
                            "exec_id": exec_id,
                            "nElo_target": point_target,
                        }
                    )
                state["next_exec_id"] = next_exec_id
                state["nElo_target"] = current_nElo_target
                save_state(state)

            # Wait for at least one future to complete
            done, not_done = wait(active_futures.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                candidate, exec_id = active_futures.pop(future)
                try:
                    loss = future.result()
                    optimizer.tell(candidate, loss)

                    nElo = -loss
                    if nElo > current_nElo_target:
                        current_nElo_target += 0.5

                    state["nElo_target"] = current_nElo_target
                    state["running"] = [
                        r for r in state["running"] if r["exec_id"] != exec_id
                    ]
                    state["completed"].append(
                        {
                            "args": candidate.args,
                            "kwargs": candidate.kwargs,
                            "loss": loss,
                            "exec_id": exec_id,
                        }
                    )
                    save_state(state)

                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    state["running"] = [
                        r for r in state["running"] if r["exec_id"] != exec_id
                    ]
                    state["nElo_target"] = current_nElo_target
                    state["next_exec_id"] = next_exec_id
                    save_state(state)

    print("Final Recommended solution:", optimizer.provide_recommendation().value)
