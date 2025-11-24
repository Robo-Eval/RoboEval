#!/usr/bin/env python3

from typing import Optional

import sys
from pathlib import Path
import tempfile
import subprocess

import mujoco
from mojo import Mojo
from PIL import Image



from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

from demonstrations.demo import Demo
from demonstrations.demo import DemoStep
from demonstrations.demo_converter import DemoConverter

from roboeval.roboeval_renderer import RoboEvalRenderer

import rich_click as click
from click_prompt import filepath_option
from rich.console import Console
from rich.progress import track

console = Console()

class CustomDemoRenderer(RoboEvalRenderer):
    def __init__(self, mojo: "Mojo"):
        super().__init__(mojo)
        self.viewer = CustomOffScreenViewer(self.model, self.data)

    def _get_viewer(self, render_mode: Optional[str] = None):
        self.viewer.make_context_current()
        return self.viewer

    def close(self):
        super().close()
        self.viewer.close()

    def set_demo_data(self, demo_info: DemoStep, actual_info: DemoStep):
        self._get_viewer().set_step_data(demo_info, actual_info)


class CustomOffScreenViewer(OffScreenViewer):

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
    ):
        super().__init__(model, data)
        self.demo_info: Optional[DemoStep] = None
        self.actual_info: Optional[DemoStep] = None

    def set_step_data(self, demo_info: DemoStep, actual_info: DemoStep):
        self.demo_info = demo_info
        self.actual_info = actual_info

    def _create_overlay(self):
        return

    def render(self):
        return super().render("rgb_array")

@click.command()
@filepath_option("--demo-path", default="~/code/RobotOlympics/data/Bimanual Panda/0118e47678174c6ea83d0ef599d11d4a.safetensors", help="Recorded demo to load")
@filepath_option("--output-path", default="./videos", prompt=False, help="Default output folder to store the rendered videos")
def cli(demo_path, output_path):
    """
    Renderes a demo to a video 
    """

    demo_path = Path(demo_path).expanduser()

    demo = Demo.from_safetensors(demo_path)

    frequency = 50
    env = demo.metadata.get_env(frequency, "human")
    robot_name = demo.metadata.environment_data.robot_name
    env_name = demo.metadata.environment_data.env_name
    demo_recorded_date = demo.metadata.date
    output_path = Path(output_path).expanduser().absolute()

    if not output_path.exists():
        output_path.mkdir()

    if not output_path.is_dir():
        console.print("[red]Error[/red] output-path is not a folder")
        sys.exit(-1)

    output_video_path = output_path / f"{env_name}_{robot_name}_{demo_recorded_date}.mp4"

    console.print(f"[blue]Reading[/blue] demo from [gray]{demo_path}[/gray]")
    console.print(f"[blue]Writing[/blue] rendered video to [gray]{output_video_path}[/gray]")

    demo = DemoConverter.decimate(demo, frequency, robot=env.robot)
    demo_renderer = CustomDemoRenderer(env.mojo)
    env.mujoco_renderer = demo_renderer

    cam = demo_renderer.viewer.cam
    cam.distance = 2.5
    cam.elevation = -25
    cam.lookat[:] = [1.0, 0, 1.0] 
    #cam.lookat[:] = [0.0, 0, 1.0] 

    # reset the env and replay the demo
    env.reset(seed=int(demo.seed))

    console.rule("Converting demo to video")

    try:
        rgb_frames = []
        for timestep in track(demo.timesteps, console=console, description="Rendering"):
            cam.azimuth = (cam.azimuth + 1) % 360
            actual_timestep = DemoStep(
                *env.step(timestep.executed_action), timestep.executed_action
            )
            demo_renderer.set_demo_data(
                demo_info=timestep, actual_info=actual_timestep
            )
            rgb_frames.append(env.render())
    except ValueError as e:
        console.print("[red]Error[/red] while rendering images: ", e)
    except KeyboardInterrupt:
        console.print("[orange]keyboard interrupt.[/orange]")
        rgb_frames.clear()
    finally:
        env.close()


    with tempfile.TemporaryDirectory() as rgb_image_output_folder:
        for i, rgb_image in track(enumerate(rgb_frames), console=console, description="Writing images"):
            Image.fromarray(rgb_image).save(f"{rgb_image_output_folder}/{i:04d}.png")
        cmd = f"ffmpeg  -r {frequency} -pattern_type glob -i '{rgb_image_output_folder}/*.png' -c:v libx264 -pix_fmt yuv420p {output_video_path}"
        subprocess.run(cmd, shell=True, check=True)
    console.rule("Done.")

if __name__ == "__main__":
    cli()

