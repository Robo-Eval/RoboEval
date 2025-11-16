import shutil
from pathlib import Path

from tqdm import tqdm

ROOT_FOLDER = Path(__file__).parent.parent
YCB_DOWNLOAD_FOLDER = ROOT_FOLDER / "resources" / "ycb"
XML_TARGET_FOLDER = ROOT_FOLDER / "roboeval" / "envs" / "xmls" / "ycb"

XML_MODEL_TEMPLATE = """
<mujoco model="ycb-{name}">
    <compiler angle="radian"/>

    <asset>
        <texture type="2d" name="ycb-object-texture" file="texture_map.png"/>
        <mesh name="ycb-object-meshfile" file="textured.obj"/>
        <material
            name="ycb-object-material"
            specular="0.2"
            shininess="0.2"
            texture="ycb-object-texture"
        />
    </asset>

    <worldbody>
        <body>
            <geom
                type="mesh"
                mesh="ycb-object-meshfile"
                material="ycb-object-material"
                mass="0.5"
                friction="1.5 0.1 0.01"
                solimp="0.95 0.99 0.001" solref="0.004 1"
                density="500" 
            />
        </body>
    </worldbody>
</mujoco>
"""


def main() -> int:
    ycb_candidates_folders = [
        ycb_folder
        for ycb_folder in YCB_DOWNLOAD_FOLDER.iterdir()
        if ycb_folder.is_dir() and (ycb_folder / "google_16k").is_dir()
    ]

    XML_TARGET_FOLDER.mkdir(exist_ok=True)

    for ycb_folder_src in tqdm(ycb_candidates_folders):
        ycb_name = ycb_folder_src.stem
        ycb_folder_dst = XML_TARGET_FOLDER / ycb_name
        ycb_folder_dst.mkdir(exist_ok=True)

        xml_model_str = XML_MODEL_TEMPLATE.format(name=ycb_name)
        with open(ycb_folder_dst / "model.xml", "w") as fhandle:
            fhandle.write(xml_model_str)

        shutil.copy(
            ycb_folder_src / "google_16k" / "textured.obj",
            ycb_folder_dst / "textured.obj",
        )
        shutil.copy(
            ycb_folder_src / "google_16k" / "textured.mtl",
            ycb_folder_dst / "textured.mtl",
        )
        shutil.copy(
            ycb_folder_src / "google_16k" / "texture_map.png",
            ycb_folder_dst / "texture_map.png"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
