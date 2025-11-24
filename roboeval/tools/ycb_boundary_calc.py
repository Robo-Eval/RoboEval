import argparse
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco as mj
import mujoco.viewer


from roboeval.envs.props.ycb import YCB_ASSETS_PATH


def compute_aabb_from_model(
    model: mj.MjModel,
) -> Tuple[np.ndarray, np.ndarray]:
    # The main collider is the first and only geom
    assert model.ngeom == 1, "YCB models should have only one collider/visual"
    assert (
        model.geom_type[0] == mj.mjtGeom.mjGEOM_MESH
    ), "YCB models should have a mesh collider/visual"

    pos = model.geom_pos[0]
    quat = model.geom_quat[0]

    mesh_id = model.geom_dataid[0]
    mesh_vert_start = model.mesh_vertadr[mesh_id]
    mesh_vert_count = model.mesh_vertnum[mesh_id]

    vertices = np.zeros((mesh_vert_count, 3))
    for i in range(mesh_vert_count):
        vertices[i] = model.mesh_vert[mesh_vert_start + i]

    # Rotate all vertices
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    vertices = rot.apply(vertices) + pos

    min_point = np.min(vertices, axis=0)
    max_point = np.max(vertices, axis=0)

    return min_point, max_point


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="003_cracker_box",
        help="The name of the YCB model to be loaded for boundary calculation",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether or not to visualize the results",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether or not to save the new AABB model",
    )
    args = parser.parse_args()

    model_filepath = YCB_ASSETS_PATH / args.model / "model.xml"
    model_spec = mj.MjSpec.from_file(str(model_filepath.resolve()))
    model = mj.MjModel.from_xml_path(str(model_filepath.resolve()))

    aabb_min, aabb_max = compute_aabb_from_model(model)
    # print(f"AABB > min = {aabb_min}, max = {aabb_max}")
    # print(f"pos: {(aabb_min + aabb_max) / 2}")

    body_spec = model_spec.worldbody.first_body()
    if len(body_spec.sites) > 0:
        # Already has a site for the bounding box
        return 0

    body_spec.add_site(
        name="boundary",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=(aabb_max - aabb_min) / 2,
        pos=(aabb_max + aabb_min) / 2,
        rgba=[0, 1, 0, 0.25],
        group=3,
    )

    model: mj.MjModel = model_spec.compile()

    if args.save:
        model_output_filepath = YCB_ASSETS_PATH / args.model / "model.xml"
        with open(model_output_filepath, "w") as fhandle:
            fhandle.write(model_spec.to_xml())

    if args.visualize:
        data = mj.MjData(model)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mj.mj_step(model, data)
                viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
