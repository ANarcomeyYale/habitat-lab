# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    CompositeSuccess,
)
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl import (  # noqa: F401.
    FixedHighLevelPolicy,
    HighLevelPolicy,
    NeuralHighLevelPolicy,
)
from habitat_baselines.rl.hrl.skills import (  # noqa: F401.
    ArtObjSkillPolicy,
    NavSkillPolicy,
    NoopSkillPolicy,
    OracleNavPolicy,
    PickSkillPolicy,
    PlaceSkillPolicy,
    ResetArmSkill,
    SkillPolicy,
    WaitSkillPolicy,
)
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.utils.common import get_num_actions

from collections import defaultdict
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExpr,
    LogicalExprType,
    LogicalQuantifierType,
)

import time
import subprocess
import itertools

@baseline_registry.register_policy
class HierarchicalPolicy(nn.Module, Policy):
    """
    :property _pddl_problem: Stores the PDDL domain information. This allows
        accessing all the possible entities, actions, and predicates. Note that
        this is not the grounded PDDL problem with truth values assigned to the
        predicates basedon the current simulator state.
    """

    _pddl_problem: PddlProblem

    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        num_envs: int,
    ):
        super().__init__()

        self._action_space = action_space
        self._num_envs: int = num_envs

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}
        self._idx_to_name: Dict[int, str] = {}

        task_spec_file = osp.join(
            full_config.habitat.task.task_spec_base_path,
            full_config.habitat.task.task_spec + ".yaml",
        )
        domain_file = full_config.habitat.task.pddl_domain_def

        self._pddl_problem = PddlProblem(
            domain_file,
            task_spec_file,
            config,
        )


        '''
        Example usage of AI-planning/pddl parser:

        from pddl.logic import Predicate, constants, variables
        from pddl.core import Domain, Problem, Action, Requirements
        from pddl.formatter import domain_to_string, problem_to_string

        # set up variables and constants
        x, y, z = variables("x y z", types=["type_1"])
        a, b, c = constants("a b c", types=["type_1"])

        # define predicates
        p1 = Predicate("p1", x, y, z)
        p2 = Predicate("p2", x, y)

        # define actions
        a1 = Action(
            "action-1",
            parameters=[x, y, z],
            precondition=p1(x, y, z) & ~p2(y, z),
            effect=p2(y, z)
        )

        # define the domain object.
        requirements = [Requirements.STRIPS, Requirements.TYPING]
        domain = Domain("my_domain",
            requirements=requirements,
            types=["type_1"],
            constants=[a, b, c],
            predicates=[p1, p2],
            actions=[a1])

        print(domain_to_string(domain))
        '''


        '''
        Example PddlDomain/Problem object attributes:

        pddl_problem.expr_types
        {'object': T:object, 'articulated_agent_type': T:articulated_agent_type, 'static_obj_type': T:static_obj_type, 'art_obj_type': T:art_obj_type, 'obj_type': T:obj_type, 'cab_type': T:cab_type, 'fridge_type': T:fridge_type, 'goal_type': T:goal_type, 'rigid_obj_type': T:rigid_obj_type}
        
        pddl_problem.constants
        {'cab_push_point_7': cab_push_point_7-T:cab_type, 'cab_push_point_6': cab_push_point_6-T:cab_type, 'cab_push_point_5': cab_push_point_5-T:cab_type, 'cab_push_point_4': cab_push_point_4-T:cab_type, 'fridge_push_point': fridge_push_point-T:fridge_type}
        
        pddl_problem.all_entities
        {'goal0|0': goal0|0-T:rigid_obj_type, 'TARGET_goal0|0': TARGET_goal0|0-T:goal_type, 'robot_0': robot_0-T:articulated_agent_type, 'cab_push_point_7': cab_push_point_7-T:cab_type, 'cab_push_point_6': cab_push_point_6-T:cab_type, 'cab_push_point_5': cab_push_point_5-T:cab_type, 'cab_push_point_4': cab_push_point_4-T:cab_type, 'fridge_push_point': fridge_push_point-T:fridge_type}
        
        pddl_problem.predicates
        {'in': <Predicate: in [[obj-T:obj_type, receptacle-T:art_obj_type]] [None]>, 'holding': <Predicate: holding [[obj-T:rigid_obj_type, robot_id-T:articulated_agent_type]] [None]>, 'not_holding': <Predicate: not_holding [[robot_id-T:articulated_agent_type]] [None]>, 'opened_cab': <Predicate: opened_cab [[cab_id-T:cab_type]] [None]>, 'closed_cab': <Predicate: closed_cab [[cab_id-T:cab_type]] [None]>, 'opened_fridge': <Predicate: opened_fridge [[fridge_id-T:fridge_type]] [None]>, 'closed_fridge': <Predicate: closed_fridge [[fridge_id-T:fridge_type]] [None]>, 'robot_at': <Predicate: robot_at [[Y-T:static_obj_type, robot_id-T:articulated_agent_type]] [None]>, 'at': <Predicate: at [[obj-T:obj_type, at_entity-T:static_obj_type]] [None]>}
        
        pddl_problem.actions
        {'nav': <Action nav ([obj-T:obj_type, robot-T:articulated_agent_type])->(None)>, 'nav_to_receptacle': <Action nav_to_receptacle ([marker-T:art_obj_type, obj-T:obj_type, robot-T:articulated_agent_type])->(None)>, 'pick': <Action pick ([obj-T:rigid_obj_type, robot-T:articulated_agent_type])->(None)>, 'place': <Action place ([place_obj-T:rigid_obj_type, obj-T:goal_type, robot-T:articulated_agent_type])->(None)>, 'open_fridge': <Action open_fridge ([fridge_id-T:fridge_type, obj-T:obj_type, robot-T:articulated_agent_type])->(None)>, 'close_fridge': <Action close_fridge ([fridge_id-T:fridge_type, obj-T:obj_type, robot-T:articulated_agent_type])->(None)>, 'open_cab': <Action open_cab ([marker-T:cab_type, obj-T:obj_type, robot-T:articulated_agent_type])->(None)>, 'close_cab': <Action close_cab ([marker-T:cab_type, obj-T:obj_type, robot-T:articulated_agent_type])->(None)>, 'nav_to_receptacle_by_name': <Action nav_to_receptacle_by_name ([marker-T:art_obj_type, robot-T:articulated_agent_type])->(None)>, 'open_fridge_by_name': <Action open_fridge_by_name ([fridge_id-T:fridge_type, robot-T:articulated_agent_type])->(None)>, 'close_fridge_by_name': <Action close_fridge_by_name ([fridge_id-T:fridge_type, robot-T:articulated_agent_type])->(None)>, 'open_cab_by_name': <Action open_cab_by_name ([marker-T:cab_type, robot-T:articulated_agent_type])->(None)>, 'close_cab_by_name': <Action close_cab_by_name ([marker-T:cab_type, robot-T:articulated_agent_type])->(None)>}
        
        note: all_entities includes only goal and target objects, not the entire set of objects that could be the goal/target
        '''


        '''
        Example plan for groceries task:

        - nav(obj0_target|0, robot_0)
        - pick(obj0_target|0, robot_0)
        - nav(TARGET_obj0_target|0, robot_0)
        - place(obj0_target|0,TARGET_obj0_target|0, robot_0)

        - nav(obj1_target|1, robot_0)
        - pick(obj1_target|1, robot_0)
        - nav(TARGET_obj1_target|1, robot_0)
        - place(obj1_target|1,TARGET_obj1_target|1, robot_0)

        - nav(obj2_target|2, robot_0)
        - pick(obj2_target|2, robot_0)
        - nav(TARGET_obj2_target|2, robot_0)
        - place(obj2_target|2,TARGET_obj2_target|2, robot_0)
        '''

       
        domain_starttime = time.time()

        types_set = {entity.expr_type for name, entity in self._pddl_problem.all_entities.items()}
        types_pddl = defaultdict(list)
        while len(types_set) > 0:
            type = types_set.pop()
            if type.parent is not None:
                types_pddl[type.parent.name].append(type.name)
                types_set.add(type.parent)

        def predicate_to_dict(pred, use_arg_values=False):
            if isinstance(pred, LogicalExpr):
                if pred._expr_type == LogicalExprType.AND:
                    return {'expr_type': 'and', 'sub_exprs': [predicate_to_dict(pred, use_arg_values) for pred in pred._sub_exprs]}
                elif pred._expr_type == LogicalExprType.NAND:
                    return {'expr_type': 'not', 'sub_exprs': 
                            [{'expr_type': 'and', 'sub_exprs': [predicate_to_dict(pred, use_arg_values) for pred in pred._sub_exprs]}]}
                elif pred._expr_type == LogicalExprType.NOT:
                    return {'expr_type': 'not', 'sub_exprs': [predicate_to_dict(pred, use_arg_values) for pred in pred._sub_exprs]}
                else:
                    raise ValueError
            else:
                args = pred._arg_values if use_arg_values else pred._args 
                return {'name': pred.name, 'params': {
                    arg.name:arg.expr_type.name for arg in args}
                }
                # TODO: should params be a list of tuples to maintain guaranteed order?
        
        predicates_pddl = []
        for name, pred in self._pddl_problem.predicates.items():
            #predicates_pddl.append({'name': pred.name, 'params': [
            #    {'name': } for param in pred._args
            #]})
            #predicates_pddl.append({'name': pred.name, 'params': {
            #    arg.name:arg.expr_type.name for arg in pred._args}
            #})
            predicates_pddl.append(predicate_to_dict(pred))

        
        def condition_to_dict(cond):
            if isinstance(cond, list):
                return [predicate_to_dict(pred, use_arg_values=True) for pred in cond]
            elif isinstance(cond, LogicalExpr):
                if cond._expr_type == LogicalExprType.AND:
                    return [predicate_to_dict(pred, use_arg_values=True) for pred in cond._sub_exprs]

        def action_to_dict(action):
            return {
                    'name': action.name,
                    'params': {arg.name:arg.expr_type.name for arg in action._params},
                    'pre_condition': condition_to_dict(action._pre_cond),
                    'post_condition': condition_to_dict(action._post_cond)
                }

        actions_pddl = []
        for name, action in self._pddl_problem.actions.items():
            actions_pddl.append(action_to_dict(action))

        constants_pddl = {name: const.expr_type.name for name, const in self._pddl_problem._constants.items()}

        domain_pddl = {
            'domain_name': 'habitat',
            'requirements': ['typing', 'strips', 'constraints', 'preferences', 'universal-preconditions', 'negative-preconditions'],
            'types': types_pddl,
            'constants': constants_pddl,
            'predicates': predicates_pddl,
            'actions': actions_pddl
        }

        def format_reqs(requirements):
            return ':' + ' :'.join(requirements)

        def format_types(types):
            return ' '.join([
                f"{' '.join(children_types)} - {parent_type}"
                for parent_type, children_types in types.items()
            ])

        def format_constants(consts):
            return ' '.join([
                f"{name} - {type}" for name, type in consts.items()
            ])

        def format_parameters(params, include_types=True):
            if include_types:
                return ' '.join([f"?{arg_name} - {arg_type}" for arg_name, arg_type in params.items()])
            else:
                return ' '.join([f"{arg_name}" if arg_name in self._pddl_problem._constants else f"?{arg_name}" for arg_name in params.keys()])

        def format_predicate(pred, include_types=True):
            #args_str = ' '.join([f"?{arg_name} - {arg_type}" for arg_name, arg_type in pred["params"].items()])
            args_str = format_parameters(pred["params"], include_types=include_types)
            return f"({pred['name']} {args_str})"

        def format_condition(cond):
            if 'expr_type' in cond:
                return f"({cond['expr_type']} {' '.join([format_condition(sub_expr) for sub_expr in cond['sub_exprs']])})"
            else:
                return format_predicate(cond, include_types=False)
        
        def format_conditions(pre_condition):
            return ' '.join([format_condition(cond) for cond in pre_condition])

        with open("pddl_workingdir/habitat_domain.pddl", mode="w") as file:
            file.write(f"(define (domain {domain_pddl['domain_name']})\n")
            
            file.write(f"\t(:requirements {format_reqs(domain_pddl['requirements'])})\n")

            file.write(f"\t(:types {format_types(domain_pddl['types'])})\n")

            file.write(f"\t(:constants {format_constants(domain_pddl['constants'])})\n")

            file.write(f"\t(:predicates\n")
            for predicate in domain_pddl["predicates"]:
                file.write(f"\t\t{format_predicate(predicate)}\n")
            file.write("\t)\n")

            for action in domain_pddl["actions"]:
                file.write(f"\t(:action {action['name']}\n")
                file.write(f"\t\t:parameters ({format_parameters(action['params'])})\n")
                file.write(f"\t\t:precondition (and {format_conditions(action['pre_condition'])})\n")
                file.write(f"\t\t:effect (and {format_conditions(action['post_condition'])})\n")
                file.write("\t)\n")

            file.write(")")
        command = 'docker cp pddl_workingdir/habitat_domain.pddl pddl_manual_dev:/root/workingdir'
        subprocess.call(command, shell=True)
        domain_endtime = time.time()

        print(f"\n\n@@@ Domain processing time = {round(domain_endtime-domain_starttime,3)} @@@\n\n")
        # @@@ Domain processing time = 0.061 @@@
        # in non debug mode: @@@ Domain processing time = 0.054 @@@

        def save_pddl_problem(problem_filename="pddl_workingdir/habitat_problem.pddl", current_state=None):

            objects = [{"name": obj.name, "type": obj.expr_type.name} for obj in self._pddl_problem._objects.values()]
            objects_pddl = defaultdict(list)
            for obj in objects:
                objects_pddl[obj["type"]].append(obj)
            
            if current_state is None:
                init_pddl = condition_to_dict(self._pddl_problem.init)
            else:
                # TODO: precompute this only once
                type_to_entities = defaultdict(set)
                for name, entity in self._pddl_problem._constants.items():
                    type = entity.expr_type
                    while type is not None:
                        type_to_entities[type.name].add(entity)
                        type = type.parent

                for name, entity in self._pddl_problem._objects.items():
                    type = entity.expr_type
                    while type is not None:
                        type_to_entities[type.name].add(entity)
                        type = type.parent

                init_preds = []
                init_preds_all = []
                for pred in current_state.predicates.values():
                    all_arg_values = [type_to_entities[arg.expr_type.name] for arg in pred._args]
                    arg_combinations = itertools.product(*all_arg_values)
                    for args in arg_combinations:
                        this_pred = pred.clone()
                        this_pred.set_param_values(args)
                        if this_pred.is_true(current_state):
                            init_preds.append(this_pred)
                        init_preds_all.append(this_pred)

                # TODO: in non-debug mode, get around sim pickle issue by extracting these true predicates instead of entire PddlProblem object 

                init_pddl = condition_to_dict(self._pddl_problem.init) + condition_to_dict(init_preds)

            goal_pddl = condition_to_dict(self._pddl_problem.goal)

            problem_pddl = {
                            "problem_name": "habitat_problem",
                            "domain_name": "habitat",
                            "objects": objects_pddl,
                            "init": init_pddl,
                            "goal": goal_pddl
                            }
            
            def format_objects(objs):
                return ' '.join([
                    f"{' '.join([type_obj['name'] for type_obj in type_objs])} - {type}"
                    for type, type_objs in objs.items()
                ])

            with open(problem_filename, mode='w') as file:

                file.write(f"(define (problem {problem_pddl['problem_name']})\n")

                file.write(f"\t(:domain {problem_pddl['domain_name']})\n")

                file.write(f"\t(:objects {format_objects(problem_pddl['objects']).replace('|','-')})\n")

                # file.write(f"\t(:init\n")
                # for predicate in problem_pddl["init"]:
                #     file.write(f"\t\t{format_predicate(predicate, include_types=False).replace('?','')}\n")
                # file.write("\t)\n")
                # # TODO: less hacky removal of question marks for pddl problem

                # file.write(f"\t(:goal\n")
                # for predicate in problem_pddl["goal"]:
                #     file.write(f"\t\t{format_predicate(predicate, include_types=False).replace('?','')}\n")
                # file.write("\t)\n")

                # TODO: better checking for empty init/goal than checking empty string
                # TODO: less hacky removal of question marks  from init and goal strings
                init_str = format_conditions(problem_pddl['init']).replace('?','')
                if init_str == '':
                    file.write(f"\t(:init )\n")
                else:
                    file.write(f"\t(:init {format_conditions(problem_pddl['init']).replace('?','').replace('|','-')})\n")
                goal_str = format_conditions(problem_pddl['goal']).replace('?','')
                if goal_str == '':
                    file.write(f"\t(:goal )\n")
                else:
                    file.write(f"\t(:goal (and {format_conditions(problem_pddl['goal']).replace('?','').replace('|','-')}))\n")

                file.write(")")

        problem_starttime = time.time()
        save_pddl_problem()
        command = 'docker cp pddl_workingdir/habitat_problem.pddl pddl_manual_dev:/root/workingdir'
        subprocess.call(command, shell=True)
        problem_endtime = time.time()
        print(f"\n\n@@@ Problem processing time (no current state) = {round(problem_endtime-problem_starttime,3)} @@@\n\n")
        # @@@ Problem processing time (no current state) = 0.05 @@@
        # in non debug mode: @@@ Problem processing time (no current state) = 0.049 @@@


        '''
        Initial pass at using AI-Planning/pddl parser repo. Hierarchical types are not supported.

        from pddl.logic import Predicate, constants, variables
        from pddl.core import Domain, Problem, Action, Requirements
        from pddl.formatter import domain_to_string, problem_to_string

        # constants = cabinet push points and refrigerator push points
        # variables = robot, objects
        

        # predicates:
        # e.g. problem.predicates['in']
        # orig 'in': args=(obj type, art obj type)
        # 

        # types
            # e.g. obj -->
                # rigid obj
                # articulated  obj
            # parser saves type as most specific type listed in python objects
            # post-processing replaces domain overall type definition with proper hierarchy

        
        obj_var = variables(
            self._pddl_problem.predicates['in']._args[0].name, 
            types=[self._pddl_problem.predicates['in']._args[0].expr_type.name])
        
        consts_names = list(self._pddl_problem.constants.keys())
        consts_types = [self._pddl_problem.constants[name].expr_type.name for name in consts_names]
        #consts_names, consts_types = zip(*[name, const.expr_type.name for name, const in self._pddl_problem.constants.items()])
        consts = constants(consts_names, types=[consts_types])

        pred_variables = []
        pred_variables2 = {}
        for name, entity in self._pddl_problem.all_entities.items():
            if name in consts_names: continue
            pred_variables.append(variables([name], [entity.expr_type.name]))
            pred_variables2[name] = variables([name], [entity.expr_type.name])

        predicates = []
        for name, pred in self._pddl_problem.predicates.items():
            pred_vars = []
            predicates.append(Predicate(name, *pred_vars))
        
        actions = []
        for name, action in self._pddl_problem.actions.items():
            pass
        '''






        skill_i = 0
        for (
            skill_name,
            skill_config,
        ) in config.hierarchical_policy.defined_skills.items():
            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config,
                observation_space,
                action_space,
                self._num_envs,
                full_config,
            )
            skill_policy.set_pddl_problem(self._pddl_problem)
            if skill_config.pddl_action_names is None:
                action_names = [skill_name]
            else:
                action_names = skill_config.pddl_action_names
            for skill_id in action_names:
                self._name_to_idx[skill_id] = skill_i
                self._idx_to_name[skill_i] = skill_id
                self._skills[skill_i] = skill_policy
                skill_i += 1

        self._cur_skills: torch.Tensor = torch.full(
            (self._num_envs,), -1, dtype=torch.long
        )

        high_level_cls = eval(
            config.hierarchical_policy.high_level_policy.name
        )
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.hierarchical_policy.high_level_policy,
            self._pddl_problem,
            num_envs,
            self._name_to_idx,
            observation_space,
            action_space,
        )
        self._high_level_policy.save_pddl_problem_fn = save_pddl_problem
        self._stop_action_idx, _ = find_action_range(
            action_space, "rearrange_stop"
        )

    def eval(self):
        pass

    def get_policy_action_space(
        self, env_action_space: spaces.Space
    ) -> spaces.Space:
        """
        Fetches the policy action space for learning. If we are learning the HL
        policy, it will return its custom action space for learning.
        """

        return self._high_level_policy.get_policy_action_space(
            env_action_space
        )

    def extract_policy_info(
        self, action_data, infos, dones
    ) -> List[Dict[str, float]]:
        ret_policy_infos = []
        for i, (info, policy_info) in enumerate(
            zip(infos, action_data.policy_info)
        ):
            cur_skill_idx = self._cur_skills[i].item()
            ret_policy_info: Dict[str, Any] = {
                "cur_skill": self._idx_to_name[cur_skill_idx],
                **policy_info,
            }

            did_skill_fail = dones[i] and not info[CompositeSuccess.cls_uuid]
            for skill_name, idx in self._name_to_idx.items():
                ret_policy_info[f"failed_skill_{skill_name}"] = (
                    did_skill_fail if idx == cur_skill_idx else 0.0
                )
            ret_policy_infos.append(ret_policy_info)

        return ret_policy_infos

    @property
    def num_recurrent_layers(self):
        if self._high_level_policy.num_recurrent_layers != 0:
            return self._high_level_policy.num_recurrent_layers
        else:
            return self._skills[0].num_recurrent_layers

    @property
    def should_load_agent_state(self):
        return False

    def parameters(self):
        return self._high_level_policy.parameters()

    def to(self, device):
        self._high_level_policy.to(device)
        for skill in self._skills.values():
            skill.to(device)

    def _broadcast_skill_ids(
        self,
        skill_ids: torch.Tensor,
        sel_dat: Dict[str, Any],
        should_adds: Optional[torch.Tensor] = None,
    ) -> Dict[int, Tuple[List[int], Dict[str, Any]]]:
        """
        Groups the information per skill. Specifically, this will return a map
        from the skill ID to the indices of the batch and the observations at
        these indices the skill is currently running for. This is used to batch
        observations per skill.
        """

        skill_to_batch: Dict[int, List[int]] = defaultdict(list)
        if should_adds is None:
            should_adds = [True for _ in range(len(skill_ids))]
        for i, (cur_skill, should_add) in enumerate(
            zip(skill_ids, should_adds)
        ):
            if should_add:
                cur_skill = cur_skill.item()
                skill_to_batch[cur_skill].append(i)
        grouped_skills = {}
        for k, v in skill_to_batch.items():
            grouped_skills[k] = (
                v,
                {dat_k: dat[v] for dat_k, dat in sel_dat.items()},
            )
        return grouped_skills

    def act(
        self,
        observations,
        bound_pddl_probs,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        masks_cpu = masks.cpu()
        log_info: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._high_level_policy.apply_mask(masks_cpu)  # type: ignore[attr-defined]

        call_high_level: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )
        bad_should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )

        hl_wants_skill_term = self._high_level_policy.get_termination(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            self._cur_skills,
            log_info,
        )
        # Initialize empty action set based on the overall action space.
        actions = torch.zeros(
            (self._num_envs, get_num_actions(self._action_space)),
            device=masks.device,
        )

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
                "actions": actions,
                "hl_wants_skill_term": hl_wants_skill_term,
            },
            # Only decide on skill termination if the episode is active.
            should_adds=masks,
        )

        # Check if skills should terminate.
        for skill_id, (batch_ids, dat) in grouped_skills.items():
            if skill_id == -1:
                # Policy has not prediced a skill yet.
                call_high_level[batch_ids] = 1.0
                continue
            # TODO: either change name of the function or assign actions somewhere
            # else. Updating actions in should_terminate is counterintuitive

            (
                call_high_level[batch_ids],
                bad_should_terminate[batch_ids],
                actions[batch_ids],
            ) = self._skills[skill_id].should_terminate(
                **dat,
                batch_idx=batch_ids,
                log_info=log_info,
                skill_name=[
                    self._idx_to_name[self._cur_skills[i].item()]
                    for i in batch_ids
                ],
            )

        # Always call high-level if the episode is over.
        call_high_level = call_high_level | (~masks_cpu).view(-1)

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate = torch.zeros(self._num_envs, dtype=torch.bool)
        hl_info: Dict[str, Any] = {}
        if call_high_level.sum() > 0:
            (
                new_skills,
                new_skill_args,
                hl_terminate,
                hl_info,
            ) = self._high_level_policy.get_next_skill(
                observations,
                bound_pddl_probs,
                rnn_hidden_states,
                prev_actions,
                masks,
                call_high_level,
                deterministic,
                log_info,
            )

            sel_grouped_skills = self._broadcast_skill_ids(
                new_skills,
                sel_dat={},
                should_adds=call_high_level,
            )

            for skill_id, (batch_ids, _) in sel_grouped_skills.items():
                self._skills[skill_id].on_enter(
                    [new_skill_args[i] for i in batch_ids],
                    batch_ids,
                    observations,
                    rnn_hidden_states,
                    prev_actions,
                )
                if "rnn_hidden_states" not in hl_info:
                    rnn_hidden_states[batch_ids] *= 0.0
                    prev_actions[batch_ids] *= 0
                elif self._skills[skill_id].has_hidden_state:
                    raise ValueError(
                        f"The code does not currently support neural LL and neural HL skills. Skill={self._skills[skill_id]}, HL={self._high_level_policy}"
                    )
            self._cur_skills = ((~call_high_level) * self._cur_skills) + (
                call_high_level * new_skills
            )

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
            },
        )
        for skill_id, (batch_ids, batch_dat) in grouped_skills.items():
            action_data = self._skills[skill_id].act(
                observations=batch_dat["observations"],
                rnn_hidden_states=batch_dat["rnn_hidden_states"],
                prev_actions=batch_dat["prev_actions"],
                masks=batch_dat["masks"],
                cur_batch_idx=batch_ids,
            )

            # LL skills are not allowed to terminate the overall episode.
            actions[batch_ids] += action_data.actions
            # Add actions from apply_postcond
            rnn_hidden_states[batch_ids] = action_data.rnn_hidden_states
        actions[:, self._stop_action_idx] = 0.0

        should_terminate = bad_should_terminate | hl_terminate
        if should_terminate.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(should_terminate):
                baselines_logger.info(
                    f"Calling stop action for batch {batch_idx}, {bad_should_terminate}, {hl_terminate}"
                )
                actions[batch_idx, self._stop_action_idx] = 1.0

        action_kwargs = {
            "rnn_hidden_states": rnn_hidden_states,
            "actions": actions,
        }
        action_kwargs.update(hl_info)

        return PolicyActionData(
            take_actions=actions,
            policy_info=log_info,
            should_inserts=call_high_level,
            **action_kwargs,
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        return self._high_level_policy.get_value(
            observations, rnn_hidden_states, prev_actions, masks
        )

    def _get_policy_components(self) -> List[nn.Module]:
        return self._high_level_policy.get_policy_components()

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        return self._high_level_policy.evaluate_actions(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
            rnn_build_seq_info,
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        **kwargs,
    ):
        return cls(
            config.habitat_baselines.rl.policy,
            config,
            observation_space,
            orig_action_space,
            config.habitat_baselines.num_environments,
        )
