#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import itertools

from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExpr,
    LogicalExprType,
    LogicalQuantifierType,
)

from habitat.tasks.rearrange.multi_task.rearrange_pddl import PddlSimInfo
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem

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

def format_objects(objs):
    return ' '.join([
        f"{' '.join([type_obj['name'] for type_obj in type_objs])} - {type}"
        for type, type_objs in objs.items()
    ])

def format_parameters(params, constants, include_types=True):
    if include_types:
        return ' '.join([f"?{arg_name} - {arg_type}" for arg_name, arg_type in params.items()])
    else:
        return ' '.join([f"{arg_name}" if arg_name in constants else f"?{arg_name}" for arg_name in params.keys()])

def format_predicate(pred, constants, include_types=True):
    #args_str = ' '.join([f"?{arg_name} - {arg_type}" for arg_name, arg_type in pred["params"].items()])
    args_str = format_parameters(pred["params"], constants, include_types=include_types)
    return f"({pred['name']} {args_str})"

def format_condition(cond, constants):
    if 'expr_type' in cond:
        return f"({cond['expr_type']} {' '.join([format_condition(sub_expr, constants) for sub_expr in cond['sub_exprs']])})"
    else:
        return format_predicate(cond, constants, include_types=False)

def format_conditions(pre_condition, constants):
    return ' '.join([format_condition(cond, constants) for cond in pre_condition])

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

def condition_to_dict(cond):
    if isinstance(cond, list):
        return [predicate_to_dict(pred, use_arg_values=True) for pred in cond]
    elif isinstance(cond, LogicalExpr):
        if cond._expr_type == LogicalExprType.AND:
            return [predicate_to_dict(pred, use_arg_values=True) for pred in cond._sub_exprs]
        elif cond._expr_type == LogicalExprType.NOT:
            # TODO: this branch of code has not been tested
            #return {'expr_type': 'not', 'sub_exprs'}
            #import pdb; pdb.set_trace()
            return predicate_to_dict(cond, use_arg_values=True)

def action_to_dict(action):
    return {
            'name': action.name,
            'params': {arg.name:arg.expr_type.name for arg in action._params},
            'pre_condition': condition_to_dict(action._pre_cond),
            'post_condition': condition_to_dict(action._post_cond)
        }

def write_pddl_domain(pddl_domain, domain_filename="pddl_workingdir/habitat_domain.pddl"):

    types_set = {entity.expr_type for name, entity in pddl_domain.all_entities.items()}
    types_pddl = defaultdict(list)
    while len(types_set) > 0:
        type = types_set.pop()
        if type.parent is not None:
            types_pddl[type.parent.name].append(type.name)
            types_set.add(type.parent)
    
    predicates_pddl = []
    for name, pred in pddl_domain.predicates.items():
        #predicates_pddl.append({'name': pred.name, 'params': [
        #    {'name': } for param in pred._args
        #]})
        #predicates_pddl.append({'name': pred.name, 'params': {
        #    arg.name:arg.expr_type.name for arg in pred._args}
        #})
        predicates_pddl.append(predicate_to_dict(pred))

    actions_pddl = []
    for name, action in pddl_domain.actions.items():
        actions_pddl.append(action_to_dict(action))

    constants_pddl = {name: const.expr_type.name for name, const in pddl_domain._constants.items()}

    domain_dict = {
        'domain_name': 'habitat',
        'requirements': ['typing', 'strips', 'constraints', 'preferences', 'universal-preconditions', 'negative-preconditions'],
        'types': types_pddl,
        'constants': constants_pddl,
        'predicates': predicates_pddl,
        'actions': actions_pddl
    }

    with open(domain_filename, mode="w") as file:
        file.write(f"(define (domain {domain_dict['domain_name']})\n")
        
        file.write(f"\t(:requirements {format_reqs(domain_dict['requirements'])})\n")

        file.write(f"\t(:types {format_types(domain_dict['types'])})\n")

        file.write(f"\t(:constants {format_constants(domain_dict['constants'])})\n")

        file.write(f"\t(:predicates\n")
        for predicate in domain_dict["predicates"]:
            file.write(f"\t\t{format_predicate(predicate, pddl_domain._constants)}\n")
        file.write("\t)\n")

        for action in domain_dict["actions"]:
            file.write(f"\t(:action {action['name']}\n")
            file.write(f"\t\t:parameters ({format_parameters(action['params'], pddl_domain._constants)})\n")
            file.write(f"\t\t:precondition (and {format_conditions(action['pre_condition'], pddl_domain._constants)})\n")
            file.write(f"\t\t:effect (and {format_conditions(action['post_condition'], pddl_domain._constants)})\n")
            file.write("\t)\n")

        file.write(")")

def get_current_predicates(pddl_problem: PddlProblem, current_state: PddlSimInfo):
    # TODO: precompute this only once
    type_to_entities = defaultdict(set)
    for name, entity in pddl_problem._constants.items():
        type = entity.expr_type
        while type is not None:
            type_to_entities[type.name].add(entity)
            type = type.parent

    for name, entity in pddl_problem._objects.items():
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
    return init_preds

def save_pddl_problem(pddl_problem, problem_filename="pddl_workingdir/habitat_problem.pddl", current_state=None):

    objects = [{"name": obj.name, "type": obj.expr_type.name} for obj in pddl_problem._objects.values()]
    objects_pddl = defaultdict(list)
    for obj in objects:
        objects_pddl[obj["type"]].append(obj)
    
    if current_state is None:
        init_pddl = condition_to_dict(pddl_problem.init)
    else:
        init_preds = get_current_predicates(pddl_problem, current_state)

        # TODO: in non-debug mode, get around sim pickle issue by extracting these true predicates instead of entire PddlProblem object 

        init_pddl = condition_to_dict(pddl_problem.init) + condition_to_dict(init_preds)

    goal_pddl = condition_to_dict(pddl_problem.goal)

    problem_dict = {
                    "problem_name": "habitat_problem",
                    "domain_name": "habitat",
                    "objects": objects_pddl,
                    "init": init_pddl,
                    "goal": goal_pddl
                    }

    with open(problem_filename, mode='w') as file:

        file.write(f"(define (problem {problem_dict['problem_name']})\n")

        file.write(f"\t(:domain {problem_dict['domain_name']})\n")

        file.write(f"\t(:objects {format_objects(problem_dict['objects']).replace('|','-')})\n")

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
        init_str = format_conditions(problem_dict['init'], pddl_problem._constants).replace('?','')
        if init_str == '':
            file.write(f"\t(:init )\n")
        else:
            file.write(f"\t(:init {format_conditions(problem_dict['init'], pddl_problem._constants).replace('?','').replace('|','-')})\n")
        goal_str = format_conditions(problem_dict['goal'], pddl_problem._constants).replace('?','')
        if goal_str == '':
            file.write(f"\t(:goal )\n")
        else:
            file.write(f"\t(:goal (and {format_conditions(problem_dict['goal'], pddl_problem._constants).replace('?','').replace('|','-')}))\n")

        file.write(")")