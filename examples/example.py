#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym

import habitat.gym  # noqa: F401
import os

import habitat

from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
)
from habitat_sim.utils import viz_utils as vut

output_path = 'test_visualizations/example_rearrange'

def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README

    '''
    <<<<<<< Updated upstream
        with gym.make("HabitatRenderPick-v0") as env:
    =======
        with habitat.Env(
    <<<<<<< Updated upstream
            config=habitat.get_config("benchmark/rearrange/pick.yaml")
    =======
            config=habitat.get_config("benchmark/rearrange/rearrange.yaml")
            #config=habitat.get_config("habitat-lab/habitat/config/habitat/task/rearrange/pddl/rearrange.yaml")
    >>>>>>> Stashed changes
        ) as env:
    >>>>>>> Stashed changes
    '''
    with habitat.Env(
        config=habitat.get_config("benchmark/rearrange/rearrange.yaml")
    ) as env:

        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        # Get metrics
        info = env.get_metrics()
        # Concatenate RGB-D observation and topdowm map into one image
        frame = observations_to_image(observations, info)

        # Remove top_down_map from metrics
        #info.pop("top_down_map")
        # Overlay numeric metrics onto frame
        frame = overlay_frame(frame, info)
        # Add fame to vis_frames
        vis_frames = [frame]

        print("Agent acting inside environment.")
        count_steps = 0
        terminal = False
        while not terminal:
            observations, reward, terminal, info = env.step(
                env.action_space.sample()
            )  # noqa: F841
            count_steps += 1

            info = env.get_metrics()
            frame = observations_to_image(observations, info)

            #info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames.append(frame)


        print("Episode finished after {} steps.".format(count_steps))

        current_episode = env.current_episode
        video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
        # Create video from images and save to disk
        images_to_video(
            vis_frames, output_path, video_name, fps=6, quality=9
        )
        vis_frames.clear()
        # Display video
        vut.display_video(f"{output_path}/{video_name}.mp4")
        print(f"Saved episode video to {output_path}/{video_name}.mp4")


if __name__ == "__main__":
    example()
