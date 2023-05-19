#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


class SimulatorObjectType(Enum):
    MOVABLE_ENTITY = "movable_entity_type"
    STATIC_RECEPTACLE_ENTITY = "static_receptacle_entity_type"
    ARTICULATED_RECEPTACLE_ENTITY = "art_receptacle_entity_type"
    GOAL_ENTITY = "goal_entity_type"
    ROBOT_ENTITY = "robot_entity_type"
    OBJ_ENTITY =  "obj_type"


def parse_func(x: str) -> Tuple[str, List[str]]:
    """
    Parses out the components of a function string.
    :returns: First element is the name of the function, second argument are the function arguments.
    """
    try:
        name = x.split("(")[0]
        args = x.split("(")[1].split(")")[0]
        args_list = args.split(",")
        args_list = [x.strip() for x in args_list]
    except IndexError as e:
        raise ValueError(f"Cannot parse '{x}'") from e

    if len(args_list) == 1 and args_list[0] == "":
        args_list = []

    return name, args_list


class ExprType:
    def __init__(self, name: str, parent: Optional["ExprType"]):
        assert isinstance(name, str)
        assert parent is None or isinstance(parent, ExprType)
        self.name = name
        self.parent = parent

    def is_subtype_of(self, other_type: "ExprType") -> bool:
        """
        If true, then `self` is compatible with `other_type` but `other_type`
        is NOT necessarily compatible with `self`. In other words, `other_type`
        is higher on the hierarchy of sub-types than `self`.
        """
        all_types = [self.name]
        parent = self.parent
        while parent is not None:
            all_types.append(parent.name)
            parent = parent.parent

        return other_type.name in all_types

    # TODO: add __eq__ operator?
    
    def __repr__(self):
        return f"T:{self.name}"


@dataclass(frozen=True)
class PddlEntity:
    name: str
    expr_type: ExprType

    def __repr__(self):
        return f"{self.name}-{self.expr_type}"

    def __eq__(self, other):
        if not isinstance(other, PddlEntity):
            return False
        return (self.name == other.name) and (
            self.expr_type.name == other.expr_type.name
        )


def do_entity_lists_match(
    to_set: List[PddlEntity], set_value: List[PddlEntity]
) -> bool:
    """
    Returns if the two predicate lists match in count and argument types.
    """

    if len(to_set) != len(set_value):
        return False
    # Check types are compatible
    return all(
        set_arg.expr_type.is_subtype_of(arg.expr_type)
        for arg, set_arg in zip(to_set, set_value)
    )


def ensure_entity_lists_match(
    to_set: List[PddlEntity], set_value: List[PddlEntity]
) -> None:
    """
    Checks if the two predicate lists match in count and argument types. If
    they don't match, an exception is thrown.
    """

    if len(to_set) != len(set_value):
        raise ValueError(
            f"Set arg values are unequal size {to_set} vs {set_value}"
        )
    # Check types are compatible
    for arg, set_arg in zip(to_set, set_value):
        if not set_arg.expr_type.is_subtype_of(arg.expr_type):
            raise ValueError(
                f"Arg type is incompatible \n{to_set}\n vs \n{set_value}"
            )


@dataclass
class PddlSimInfo:
    obj_ids: Dict[str, int]
    target_ids: Dict[str, int]
    #episode_locs: Dict[str, int]
    art_handles: Dict[str, int]
    marker_handles: Dict[str, MarkerInfo]
    robot_ids: Dict[str, int]

    sim: RearrangeSim #can't pickle
    dataset: RearrangeDatasetV0
    env: RearrangeTask # can't pickle due to SceneNode
    episode: Episode
    obj_thresh: float
    art_thresh: float
    robot_at_thresh: float
    expr_types: Dict[str, ExprType]
    predicates: Dict[str, Any]
    all_entities: Dict[str, Any]
    receptacles: Dict[str, mn.Range3D]

    num_spawn_attempts: int
    physics_stability_steps: int
    recep_place_shrink_factor: float

    def get_predicate(self, pred_name: str):
        return self.predicates[pred_name]

    def check_type_matches(self, entity: PddlEntity, match_name: str) -> bool:
        return entity.expr_type.is_subtype_of(self.expr_types[match_name])

    def get_entity_pos(self, entity: PddlEntity) -> np.ndarray:
        ename = entity.name
        if self.check_type_matches(
            entity, SimulatorObjectType.ROBOT_ENTITY.value
        ):
            robot_id = self.robot_ids[ename]
            return self.sim.get_agent_data(robot_id).robot.base_pos
        if self.check_type_matches(
            entity, SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
        ):
            marker_info = self.marker_handles[ename]
            return marker_info.get_current_position()
        if self.check_type_matches(
            entity, SimulatorObjectType.GOAL_ENTITY.value
        ):
            if ename in self.target_ids:
                idx = self.target_ids[ename]
                targ_idxs, pos_targs = self.sim.get_targets()
                rel_idx = targ_idxs.tolist().index(idx)
                return pos_targs[rel_idx]
            elif ename == "START":
                pos = self.episode.start_position
                return np.array(pos)
        if self.check_type_matches(
            entity, SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
        ):
            recep = self.receptacles[ename]
            return np.array(recep.center())
        if self.check_type_matches(
            entity, SimulatorObjectType.MOVABLE_ENTITY.value
        ):
            rom = self.sim.get_rigid_object_manager()
            idx = self.obj_ids[ename]
            abs_obj_id = self.sim.scene_obj_ids[idx]
            cur_pos = rom.get_object_by_id(
                abs_obj_id
            ).transformation.translation
            return cur_pos
        # entity obj-T:obj_type with name 'obj' fails this elif cascade
        else:
            raise ValueError()
        # elif self.check_type_matches(entity, GOAL_TYPE):
        #     if ename in self.target_ids:
        #         idx = self.target_ids[ename]
        #         targ_idxs, pos_targs = self.sim.get_targets()
        #         rel_idx = targ_idxs.tolist().index(idx)
        #         return pos_targs[rel_idx]
        #     elif ename == "START":
        #         pos = self.episode.start_position
        #         return np.array(pos)
        # elif self.check_type_matches(entity, RIGID_OBJ_TYPE):
        #     rom = self.sim.get_rigid_object_manager()
        #     idx = self.obj_ids[ename]
        #     abs_obj_id = self.sim.scene_obj_ids[idx]
        #     cur_pos = rom.get_object_by_id(
        #         abs_obj_id
        #     ).transformation.translation
        #     return cur_pos
        # entity obj-T:obj_type with name 'obj' fails this elif cascade

    # TODO: is this deprecated now?
    def search_for_entity_any(self, entity: PddlEntity):
        ename = entity.name
        if self.check_type_matches(entity, articulated_agent_type):
            return self.robot_ids[ename]
        elif self.check_type_matches(entity, ART_OBJ_TYPE):
            return self.marker_handles[ename]
        elif self.check_type_matches(entity, GOAL_TYPE):
            if ename == "START":
                return ename
            else:
                return self.target_ids[ename]
        elif self.check_type_matches(entity, RIGID_OBJ_TYPE):
            return self.obj_ids[ename]
        else:
            raise ValueError()

    def search_for_entity(
        self, entity: PddlEntity
    ) -> Union[int, str, MarkerInfo, mn.Range3D]:
        ename = entity.name

        if self.check_type_matches(
            entity, SimulatorObjectType.ROBOT_ENTITY.value
        ):
            return self.robot_ids[ename]
        elif self.check_type_matches(
            entity, SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
        ):
            return self.marker_handles[ename]
        elif self.check_type_matches(
            entity, SimulatorObjectType.GOAL_ENTITY.value
        ):
            if ename == "START":
                return ename
            else:
                return self.target_ids[ename]
        elif self.check_type_matches(
            entity, SimulatorObjectType.MOVABLE_ENTITY.value
        ):
            return self.obj_ids[ename]
        elif self.check_type_matches(
            entity, SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
        ):
            asset_name = ename.split("_:")[0]
            return self.receptacles[asset_name]
        # elif expected_type == GOAL_TYPE:
        #     if ename == "START":
        #         return ename
        #     else:
        #         return self.target_ids[ename]
        # elif expected_type == RIGID_OBJ_TYPE:
        #     return self.obj_ids[ename]
        # elif expected_type == OBJ_TYPE:
        #     if ename in self.obj_ids and ename in self.target_ids:
        #         raise ValueError("Entity is both an object and a target")
        #     elif ename in self.obj_ids:
        #         return self.obj_ids[ename]
        #     elif ename in self.target_ids:
        #         return self.target_ids[ename]
        #     else:
        #         raise ValueError("Entity is neither an object nor a target")
        else:
            raise ValueError(f"No type match for {entity}")
