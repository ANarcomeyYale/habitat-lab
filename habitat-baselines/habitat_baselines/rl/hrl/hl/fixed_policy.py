# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple
import os
import json

import torch

from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat.tasks.rearrange.multi_task.pddl_conversion_utils import save_pddl_problem, DOCKER_NAME, pddl_state_to_dict
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy

import subprocess
import time

EXECUTE_FIXED_PLAN = False
COMPARE_FIXED_PLAN = False
MAX_HIGH_LEVEL_ACTIONS = 30

# TODO: migrate online plan to a new policy class instead of fixed policy
# TODO: migrate save pddl domain and problem functions, and parsing functions to a new file
    # for online planning

# TODO: tensorboard logging of the high level plan, success of failure of each action, success of the task
# TODO: speeding things up
    # precompute where possible

class FixedHighLevelPolicy(HighLevelPolicy):
    """
    Executes a fixed sequence of high-level actions as specified by the
    `solution` field of the PDDL problem file.
    :property _solution_actions: List of tuples were first tuple element is the
        action name and the second is the action arguments.
    """

    _solution_actions: List[Tuple[str, List[str]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._solution_actions = self._parse_solution_actions(
            self._pddl_prob.solution
        )

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self._reset_arm = torch.ones(self._num_envs, dtype=torch.bool)

        self.last_problem_starttime = None
        self.last_highlevel_starttime = None

        self.fixed_plan_actions = [[] for _ in range(self._num_envs)]
        self.online_plan_actions = [[] for _ in range(self._num_envs)]
        self.trajectory = [[] for _ in range(self._num_envs)]

    def _parse_solution_actions(self, solution):
        solution_actions = []
        for i, hl_action in enumerate(solution):
            sol_action = (
                hl_action.name,
                [x.name for x in hl_action.param_values],
            )
            solution_actions.append(sol_action)

            if self._config.add_arm_rest and i < (len(solution) - 1):
                solution_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        solution_actions.append(parse_func("wait(30)"))

        return solution_actions

    def _parse_plan_actions(self, plan_file="pddl_workingdir/plan.txt"):
        """
        Example plan.txt format:
        ;;;; Solution Found
        ; States evaluated: 37
        ; Cost: 0.002
        ; Time 0.06
        0.000: (nav start goal0-0 robot_0)  [0.001]
        0.001: (pick goal0-0 robot_0)  [0.001]
        0.002: (place goal0-0 target_goal0-0 robot_0)  [0.001]
        """

        with open(plan_file, mode='r') as file:
            plan_lines = [line.rstrip() for line in file]
        
        plan = []
        plan_prefix_start = False
        plan_start = False
        num_prefix_lines = 0
        for line in plan_lines:
            if num_prefix_lines >= 4:
                plan_start = True
                plan_prefix_start = False
            if plan_start:
                plan_step = line[line.find("(")+1 : line.find(")")]
                plan.append(plan_step)
            
            if ";;;;" in line: plan_prefix_start = True
            if plan_prefix_start:
                num_prefix_lines += 1
                continue

        plan_actions = []
        for plan_step in plan:
            plan_step = plan_step.replace('-','|')
            # TODO: better hack than replacing | with - to satisfy pddl character restrictions
            
            plan_step = plan_step.replace('target_obj0_target|0', 'TARGET_obj0_target|0')
            plan_step = plan_step.replace('target_obj1_target|1', 'TARGET_obj1_target|1')
            plan_step = plan_step.replace('target_obj2_target|2', 'TARGET_obj2_target|2')
            plan_step = plan_step.replace('target_goal', 'TARGET_goal')
            plan_step = plan_step.replace('start', 'START')
            # TODO: better hack than capitalizing target goal. Optic planner forces lowercase.
                # Ideally, actually compare argument entities with list of known entities and convert if there's
                # a match for all but - to | or capitalization
            action_name, params = plan_step.split(' ')[0], plan_step.split(' ')[1:]
            plan_actions.append((action_name, params))

        return plan_actions

    def apply_mask(self, mask):
        """
        Apply the given mask to the next skill index.

        Args:
            mask: Binary mask of shape (num_envs, ) to be applied to the next
                skill index.
        """
        # TODO: vectorize this for efficiency
        self._next_sol_idxs *= mask.cpu().view(-1)
        for i, not_done in enumerate(mask):
            if not_done == False:
                self.fixed_plan_actions[i] = []
                self.online_plan_actions[i] = []
                self.trajectory[i] = []

    def _get_next_sol_idx(self, batch_idx, immediate_end):
        """
        Get the next index to be used from the list of solution actions.

        Args:
            batch_idx: The index of the current environment.

        Returns:
            The next index to be used from the list of solution actions.
        """
        if self._next_sol_idxs[batch_idx] >= len(self._solution_actions):
            baselines_logger.info(
                f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
            )
            immediate_end[batch_idx] = True
            return len(self._solution_actions) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def save_plan_actions(self, trajectory, episode_id, preference_id, filename):

        steps_json = []
        for cost, action, state in trajectory:
            steps_json.append({'cost': cost, 'action':action, 'state':state})

        # TODO: later on, integrate the trajectories for same episode diff prefs into the same dict
        # do this as a post-processing step once the raw trajectories are encoded
        traj_json = {episode_id: {preference_id: steps_json}}
        json.dump(traj_json, open(filename, 'w'))
    
    def get_next_skill(
        self,
        observations,
        bound_pddl_probs,
        episodes_info,
        planner_config,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):

        highlevel_starttime = time.time()
        if self.last_highlevel_starttime is not None:
            print(f"\n\n@@@ Time between high level planner runs = {round(highlevel_starttime-self.last_highlevel_starttime,3)} @@@\n\n")
        self.last_highlevel_starttime = highlevel_starttime
        # debugging mode: highly variable from ~13 seconds to 5 seconds or even 0.6 seconds
        # non debugging mode: ~15 seconds
        # Note: even when returning the fixed plan, debugging mode is finishing faster. This is odd

        if COMPARE_FIXED_PLAN:
            next_skill = torch.zeros(self._num_envs)
            skill_args_data = [None for _ in range(self._num_envs)]
            immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
            for batch_idx, should_plan in enumerate(plan_masks):
                if should_plan == 1.0:
                    use_idx = self._get_next_sol_idx(batch_idx, immediate_end)

                    skill_name, skill_args = self._solution_actions[use_idx]
                    baselines_logger.info(
                        f"Got next element of the plan with {skill_name}, {skill_args}"
                    )
                    if skill_name not in self._skill_name_to_idx:
                        raise ValueError(
                            f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                        )
                    next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                    skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                    self._next_sol_idxs[batch_idx] += 1
                    self.fixed_plan_actions[batch_idx].append((skill_name, skill_args))

                    cost = 1
                    action = (skill_name, skill_args)
                    state = pddl_state_to_dict(bound_pddl_probs[batch_idx], bound_pddl_probs[batch_idx]._sim_info)
                    self.trajectory[batch_idx].append((cost, action, state))

                    if immediate_end[batch_idx]:
                        ep_info =  episodes_info[batch_idx]
                        ep_id = f'ep{ep_info.episode_id}_scene{os.path.splitext(os.path.basename(ep_info.scene_id))[0]}'
                        #ep_id = f'ep{ep_info.episode_id}
                        #pref_id = planner_config.pref_id
                        filename = str(f"trajectories/{ep_id}_{pref_id}.json")
                        self.save_plan_actions(self.trajectory[batch_idx], ep_id, pref_id, filename)

            fixed_plan = next_skill, skill_args_data, immediate_end, {}
            #return fixed_plan

        if EXECUTE_FIXED_PLAN:
            return fixed_plan

        next_skill = torch.zeros(self._num_envs)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                problem_starttime = time.time()
                # if self.last_problem_starttime is not None:
                #     print(f"\n\n@@@ Time between high level planner runs = {round(problem_starttime-self.last_problem_starttime,3)} @@@\n\n")
                # self.last_problem_starttime = problem_starttime
                # @@@ Time between high level planner runs = 0.569 @@@
                # @@@ Time between high level planner runs = 2.547 @@@
                # highly variable depending on the action and how long it takes

                preferences = json.load(open(planner_config.pref_filename, 'r'))
                
                #import pdb; pdb.set_trace()
                save_pddl_problem(
                    bound_pddl_probs[batch_idx],
                    problem_filename=f"pddl_workingdir/habitat_problem_{batch_idx}.pddl", 
                    current_state=bound_pddl_probs[batch_idx]._sim_info,
                    preferences=preferences)
                
                # TODO: replace subprocess calls with Docker API calls for greater safety/security
                #command = f'docker cp pddl_workingdir/habitat_problem_{batch_idx}.pddl {DOCKER_NAME}:/root/workingdir'
                #subprocess.call(command, shell=True)
                problem_endtime = time.time()
                print(f"\n\n@@@ Problem processing time (with current state) = {round(problem_endtime-problem_starttime,3)} @@@\n\n")
                # @@@ Problem processing time (with current state) = 0.122 @@@
                print("Problem.pddl:")
                subprocess.call(f"cat pddl_workingdir/habitat_problem_{batch_idx}.pddl", shell=True)

                planning_starttime = time.time()
                #command = 'docker exec --workdir /root/workingdir pddl_manual_dev optic habitat_domain.pddl habitat_problem.pddl >> plan.txt'
                #command = 'python3 -c "import planutils; planutils.main()" run optic habitat_domain.pddl habitat_problem.pddl >> plan.txt'
                command = f'docker exec --workdir /root/workingdir {DOCKER_NAME} python3 -c "import planutils; planutils.main()" run optic habitat_domain.pddl habitat_problem_{batch_idx}.pddl >> pddl_workingdir/plan_{batch_idx}.txt'
                subprocess.call(f'rm pddl_workingdir/plan_{batch_idx}.txt', shell=True)
                subprocess.call(command, shell=True)
                #command = f'docker cp pddl_manual_dev:/root/workingdir/plan_{batch_idx}.txt pddl_workingdir'
                #subprocess.call(command, shell=True)
                # TODO: output piping saves output locally instead of on docker container. Keep it this way?
                

                plan_actions = self._parse_plan_actions(plan_file=f"pddl_workingdir/plan_{batch_idx}.txt")
                print("\nplan actions = ", plan_actions, "\n")
                print("fixed plan = ", self._solution_actions)
                if len(plan_actions) == 0:
                    # Terminate episode if the planner shows no more actions to take
                    skill_name, skill_args = parse_func("wait(30)")
                    immediate_end[batch_idx] = True
                elif len(self.online_plan_actions[batch_idx]) > MAX_HIGH_LEVEL_ACTIONS:
                    # Terminate episode if the policy has executed more than <threshold> actions
                    skill_name, skill_args = parse_func("wait(30)")
                    immediate_end[batch_idx] = True
                else:
                    # Execute high level action from the planner
                    if self._config.add_arm_rest:
                        if self._reset_arm[batch_idx]:
                            skill_name, skill_args = plan_actions[0]
                        else:
                            skill_name, skill_args = parse_func("reset_arm(0)")
                        self._reset_arm[batch_idx] = not self._reset_arm[batch_idx]
                    else:
                        skill_name, skill_args = plan_actions[0]
                
                planning_endtime = time.time()
                print(f"\n\n@@@ Planning time = {round(planning_endtime-planning_starttime,3)} @@@\n\n")
                # @@@ Planning time = 0.41 @@@
                
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                self.online_plan_actions[batch_idx].append((skill_name, skill_args))
                cost = 1
                action = (skill_name, skill_args)
                state = pddl_state_to_dict(bound_pddl_probs[batch_idx], bound_pddl_probs[batch_idx]._sim_info)
                self.trajectory[batch_idx].append((cost, action, state))

                if immediate_end[batch_idx]:
                    ep_info =  episodes_info[batch_idx]
                    ep_id = f'ep{ep_info.episode_id}_scene{os.path.splitext(os.path.basename(ep_info.scene_id))[0]}'
                    #ep_id = f'ep{ep_info.episode_id}
                    pref_id = list(preferences.keys())[0]
                    filename = str(f"trajectories/{ep_id}_{pref_id}.json")
                    self.save_plan_actions(self.trajectory[batch_idx], ep_id, pref_id, filename)

        print("\n\nOnline plan so far = ", self.online_plan_actions)
        print("Fixed plan so far = ", self.fixed_plan_actions)

        
        return next_skill, skill_args_data, immediate_end, {}
