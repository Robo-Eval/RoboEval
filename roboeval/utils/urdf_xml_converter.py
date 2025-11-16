import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

"""
This helper script converts a URDF file (and its OBJ/MTL assets) into an MJCF file,
placing them into 'envs/xmls/props/<model_name>/<model_name>.xml' plus an 'assets' dir
with all meshes. 
"""

def get_model_name_from_urdf(urdf_path: Path) -> str:
    """
    Parse the URDF file and return the 'name' attribute of the root <robot> element.
    If not found, return 'unknown_model'.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    return root.attrib.get("name", "unknown_model")


def convert_urdf_to_mjcf(urdf_path: Path, output_xml: Path):
    """
    Convert URDF to an MJCF XML file using MuJoCo's Python API.
    The current mujoco.mj_saveLastXML signature is:
        mj_saveLastXML(filename: str, model: MjModel)
    so we only pass two arguments.
    """
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(output_xml), model)
    print(f"Converted URDF '{urdf_path}' to MJCF: {output_xml}")


def remove_existing_compiler_and_insert_new(
    mjcf_path: Path,
    mesh_dir: str,
    texture_dir: str,
    autolimits: bool,
    final_xml: Path
):
    """
    1) Load the MJCF at 'mjcf_path' with xml.etree.ElementTree.
    2) Remove any existing <compiler> element.
    3) Insert a new <compiler> with specified attributes.
    4) Set all mesh elements to have scale="0.2 0.2 0.2".
    5) Write to 'final_xml'.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Remove any existing <compiler>
    existing_compiler = root.find("compiler")
    if existing_compiler is not None:
        root.remove(existing_compiler)

    # Create new compiler element
    new_compiler = ET.Element("compiler")
    new_compiler.set("angle", "radian")
    new_compiler.set("meshdir", mesh_dir)
    new_compiler.set("texturedir", texture_dir)
    new_compiler.set("autolimits", "true" if autolimits else "false")

    # Insert as first child
    root.insert(0, new_compiler)

    # For each <mesh> in the MJCF, add scale attribute
    for mesh_elem in root.findall(".//mesh"):
        mesh_elem.set("scale", "0.2 0.2 0.2")

    # Write final file
    tree.write(final_xml, encoding="utf-8", xml_declaration=True)
    print(f"Final MJCF with <compiler> tag written to: {final_xml}")


def scale_positions_in_xml(xml_path: Path, scale_factor: float):
    """
    Parse the XML at 'xml_path' and scale all 'pos' attribute values by scale_factor.
    This ensures that body, joint, and geom position values match the mesh scaling.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterate over all elements that have a 'pos' attribute
    for elem in root.iter():
        if "pos" in elem.attrib:
            try:
                pos_values = [float(val) for val in elem.attrib["pos"].split()]
                scaled_values = [val * scale_factor for val in pos_values]
                elem.attrib["pos"] = " ".join(f"{v:g}" for v in scaled_values)
                
            except Exception as e:
                print(f"Error scaling pos for element {elem.tag}: {e}")
        
        if "range" in elem.attrib:
            try: 
                range_values = [float(val) for val in elem.attrib["range"].split()]
                scaled_range = [val * scale_factor for val in range_values]
                elem.attrib["range"] = " ".join(f"{v:g}" for v in scaled_range)
            except Exception as e: 
                print(f"Error with scaling range values")

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Updated positions in {xml_path} with scale factor {scale_factor}")


def copy_assets_except_urdf(urdf_path: Path, target_assets_dir: Path):
    """
    Copy all files from the URDF's directory (besides the URDF itself)
    into 'target_assets_dir'.
    """
    source_dir = urdf_path.parent
    target_assets_dir.mkdir(parents=True, exist_ok=True)
    allowable_file_ext = [".obj", ".png", ".jpg", ".stl", ".mtl", ".obj.convex.stl"]

    for item in source_dir.iterdir():
        if item.is_file():
            # Skip the URDF itself
            if item == urdf_path or item.suffix not in allowable_file_ext:
                continue
            # Copy other files
            shutil.copy2(item, target_assets_dir)
            print(f"Copied {item.name} -> {target_assets_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py /path/to/mobility.urdf")
        sys.exit(1)

    urdf_file_path = Path(sys.argv[1])
    if not urdf_file_path.is_file():
        print(f"URDF file not found: {urdf_file_path}")
        sys.exit(1)

    # 1) Extract model name from URDF
    model_name = get_model_name_from_urdf(urdf_file_path)
    print(f"Model name from URDF: {model_name}")

    # 2) Determine paths relative to the script's location
    script_path = Path(__file__).resolve()
    one_dir_up = script_path.parent.parent
    base_dir = one_dir_up / "envs" / "xmls" / "props"
    model_dir = base_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/using directory: {model_dir}")

    # The path for the final MJCF inside that folder:
    final_xml_path = model_dir / f"{model_name}.xml"

    # 3) Convert URDF to a temporary MJCF in the same folder
    temp_xml_path = model_dir / "converted_mjcf.xml"
    convert_urdf_to_mjcf(urdf_file_path, temp_xml_path)

    # 4) Remove any <compiler>, insert new <compiler meshdir='assets' ...>, write final MJCF
    remove_existing_compiler_and_insert_new(
        mjcf_path=temp_xml_path,
        mesh_dir="assets",
        texture_dir="assets",
        autolimits=True,
        final_xml=final_xml_path
    )

    # 5) Scale all 'pos' attributes in the final MJCF by 0.2 to match the mesh scaling
    scale_positions_in_xml(final_xml_path, 0.2)

    # 6) Copy all files except URDF into "model_dir/assets"
    assets_dir = model_dir / "assets"
    copy_assets_except_urdf(urdf_file_path, assets_dir)

    # Optionally remove the temporary XML file if not needed
    temp_xml_path.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()