"""Different pickable items from garage environment."""
from pathlib import Path

from dm_control import mjcf
from mujoco_utils import mjcf_utils

from roboeval.const import ASSETS_PATH
from abc import ABC
from typing import Any, Dict, List, Optional
import numpy as np
from mojo.elements import Body, MujocoElement, Joint, Site

from roboeval.envs.props.prop import CollidableProp, KinematicProp
from roboeval.utils.physics_utils import set_joint_position, get_joint_position



class Drill(KinematicProp):
    """Drill."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/drill/drill.xml"

class Screwdriver(KinematicProp):
    """Screwdriver."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/screwdriver/Screwdriver.xml"

class Nut(KinematicProp):
    """Nut."""

    _BOLT_SITE = "bolt_site"
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/nut_bolt/nut.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
        self.bolt_site = Site.get(self._mojo, self._BOLT_SITE, object)
        
    
class Bolt(KinematicProp):
    """Bolt."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/nut_bolt/bolt.xml"
    
class Toolholder(CollidableProp):
    """Toolholder."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/tool_holder/tool_holder.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model) 
        self.obj_body = Body.get(self._mojo, "surface", object).geoms[-1]  
            

class Wallstand(CollidableProp):
    """Wall mount."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/wall_stand/wall_stand.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)  
        self.obj_body = Body.get(self._mojo, "surface", object).geoms[-1] 
            
class Workbench(CollidableProp):
    """Work Bench."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/work_bench/work_bench.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)  
        self.counter = Body.get(self._mojo, "counter", object).geoms[-1] 
    
class Valve(KinematicProp):
    '''
    Define an articulated valve object
    '''
    
    _VALVE = "valve"
    
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / self.asset_path

    def _post_init(self):
        self._joints = [i for i in self.body.joints if i.mjcf.tag != 'freejoint']

    def set_state(self, state: Optional[float] = None, target_state: Optional[Dict[str, float]] = None):
        """
        Set the state of the joints.

        Args:
            state (Optional[float]): A single normalized value to set all joints to. If None, `target_state` must be provided.
            target_state (Optional[Dict[str, float]]): A dictionary mapping joint names to their desired normalized values.
        """
        if not self._joints:
            if state is not None or (target_state and len(target_state) > 0):
                print("[WARNING] set_state called but no joints available.")
            return

        if target_state:
            for joint_name, value in target_state.items():
                # Fetch the joint reference
                joint = (Joint.get(self._mojo, joint_name, self.body))
                print(f'{joint_name} is being loaded')
                if joint and hasattr(joint.mjcf, 'range'):
                    set_joint_position(joint, value, True)
                else:
                    print(f"[WARNING] Joint '{joint_name}' not found in {self.asset_path}")
        elif state is not None:
            for joint in self.all_joints:
                if joint.range is None:
                    continue

                parent = joint.root.model
                print(joint.root.model)
                joint_name = f'{parent}/{joint.name}'

                joint_obj = (Joint.get(self._mojo, joint_name, self.body))
                if joint_obj and hasattr(joint_obj.mjcf, 'range') and len(joint.range) > 0:
                    print(joint.range)
                    print(f'{joint_name} is being loaded')
                    set_joint_position(joint_obj, state, True)
        else:
            print("[WARNING] set_state called with neither 'state' nor 'target_state'. Nothing to do.")

    def get_state(self) -> np.ndarray[float]:
        """Get normalized state of joints."""
        if not self._joints:
            return np.array([])
        return np.array([get_joint_position(joint, True) for joint in self._joints])

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self.asset_path =  kwargs.get("asset_path", "props/garage/handwheel_valve/handwheel_valve.xml") # specify the customizable asset path as input
        self.bodies_to_disable: list[str] = kwargs.get("bodies_to_disable", [])
        self.target_joints: list[str] = kwargs.get("target_joints", [])

    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)  
        self.valve = Body.get(self._mojo, self._VALVE, object).geoms[-1]
 
        # Disable (remove) specific bodies if requested
        for body_name in self.bodies_to_disable:
            body_elem = model.find("body", body_name)
            if body_elem is not None:
                body_elem.remove()

        # Collect references to *all* bodies, joints, and sites
        self.all_bodies = model.find_all("body")
        self.all_joints = model.find_all("joint")
        self.all_sites  = model.find_all("site")
            
class Lever(KinematicProp):
    '''
    Define an articulated lever object
    '''
    
    
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / self.asset_path

    def _post_init(self):
        self._joints = [i for i in self.body.joints if i.mjcf.tag != 'freejoint']

    def set_state(self, state: Optional[float] = None, target_state: Optional[Dict[str, float]] = None):
        """
        Set the state of the joints.

        Args:
            state (Optional[float]): A single normalized value to set all joints to. If None, `target_state` must be provided.
            target_state (Optional[Dict[str, float]]): A dictionary mapping joint names to their desired normalized values.
        """
        if not self._joints:
            if state is not None or (target_state and len(target_state) > 0):
                print("[WARNING] set_state called but no joints available.")
            return

        if target_state:
            for joint_name, value in target_state.items():
                # Fetch the joint reference
                joint = (Joint.get(self._mojo, joint_name, self.body))
                print(f'{joint_name} is being loaded')
                if joint and hasattr(joint.mjcf, 'range'):
                    set_joint_position(joint, value, True)
                else:
                    print(f"[WARNING] Joint '{joint_name}' not found in {self.asset_path}")
        elif state is not None:
            for joint in self.all_joints:
                if joint.range is None:
                    continue

                parent = joint.root.model
                print(joint.root.model)
                joint_name = f'{parent}/{joint.name}'

                joint_obj = (Joint.get(self._mojo, joint_name, self.body))
                if joint_obj and hasattr(joint_obj.mjcf, 'range') and len(joint.range) > 0:
                    print(joint.range)
                    print(f'{joint_name} is being loaded')
                    set_joint_position(joint_obj, state, True)
        else:
            print("[WARNING] set_state called with neither 'state' nor 'target_state'. Nothing to do.")

    def get_state(self) -> np.ndarray[float]:
        """Get normalized state of joints."""
        if not self._joints:
            return np.array([])
        return np.array([get_joint_position(joint, True) for joint in self._joints])

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self.asset_path =  kwargs.get("asset_path", "props/garage/quarter_valve/quarter_valve.xml") # specify the customizable asset path as input
        self.bodies_to_disable: list[str] = kwargs.get("bodies_to_disable", [])
        self.target_joints: list[str] = kwargs.get("target_joints", [])

    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   

        # Disable (remove) specific bodies if requested
        for body_name in self.bodies_to_disable:
            body_elem = model.find("body", body_name)
            if body_elem is not None:
                body_elem.remove()

        # Collect references to *all* bodies, joints, and sites
        self.all_bodies = model.find_all("body")
        self.all_joints = model.find_all("joint")
        self.all_sites  = model.find_all("site")

        
    
"""Dumbbell Rack Assets"""
class DumbbellRackBase(KinematicProp):
    """Base of Dumbbell Rack."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/dumbbell_rack/base/base.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)  
        self.top_rack = Body.get(self._mojo, "top-rack", object).geoms[-1]
         
class Weight1(KinematicProp):
    """Weight 1."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/dumbbell_rack/weight_1/weight_1.xml"

class Weight2(KinematicProp):
    """Weight 2."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/dumbbell_rack/weight_2/weight_2.xml"

class Weight3(KinematicProp):
    """Weight 3."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/dumbbell_rack/weight_3/weight_3.xml"

class Weight4(KinematicProp):
    """Weight 4."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/dumbbell_rack/weight_4/weight_4.xml"

class Weight5(KinematicProp):
    """Weight 5."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/dumbbell_rack/weight_5/weight_5.xml"

class Weight6(KinematicProp):
    """Weight 6."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/dumbbell_rack/weight_6/weight_6.xml"

class Weight7(KinematicProp):
    """Weight 7."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/dumbbell_rack/weight_7/weight_7.xml"

"""Montessori Assets"""
class MontessoriBase(KinematicProp):
    """Montessori Base."""
    _CIRCLE_SITE = "circle_site"
    _CUBE_SITE = "cube_site"
    _TRIANGLE_SITE = "triangle_site"
    _PENTAGON_SITE = "pentagon_site"
    _RECTANGLE_SITE = "rectangle_site"
    
    _CIRCLE = "circle"
    _RECTANGLE = "rectangle"
    _CUBE = "cube"
    _PENTAGON = "pentagon"
    _TRIANGLE = "triangle"
    
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/montessori/base/base.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
        self.circle_site = Site.get(self._mojo, self._CIRCLE_SITE, object)
        self.cube_site = Site.get(self._mojo, self._CUBE_SITE, object)
        self.pentagon_site = Site.get(self._mojo, self._PENTAGON_SITE, object)
        self.rectangle_site = Site.get(self._mojo, self._RECTANGLE_SITE, object)
        self.triangle_site = Site.get(self._mojo, self._TRIANGLE_SITE, object)
        self.sites = [self.circle_site, self.cube_site, self.pentagon_site, self.rectangle_site, self.triangle_site]
        
        # Get geoms for each hole target
        self.circle_geoms = [geom for geom in Body.get(self._mojo, self._CIRCLE, object).geoms]
        self.cube_geoms = [geom for geom in Body.get(self._mojo, self._CUBE, object).geoms]
        self.pentagon_geoms = [geom for geom in Body.get(self._mojo, self._PENTAGON, object).geoms]
        self.rectangle_geoms = [geom for geom in Body.get(self._mojo, self._RECTANGLE, object).geoms]
        self.triangle_geoms = [geom for geom in Body.get(self._mojo, self._TRIANGLE, object).geoms]


class MontessoriCircle(KinematicProp):
    """Montessori Circle."""
    _HOLE_SITE = "hole_site"
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/montessori/circle/circle.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
        # self.hole_site = Site.get(self._mojo, self._HOLE_SITE, object)
        # self.sites  = [self.hole_site]
        
        self.holes = [geom for geom in Body.get(self._mojo, self._HOLE_SITE, object).geoms]


class MontessoriCube(KinematicProp):
    """Montessori Cube."""
    _HOLE_SITE_1 = "hole_site_1"
    _HOLE_SITE_2 = "hole_site_2"
    _HOLE_SITE_3 = "hole_site_3"
    _HOLE_SITE_4 = "hole_site_4"
    
    _HOLE = "hole_site"


    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/montessori/cube/cube.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
        # self.hole_site_1 = Site.get(self._mojo, self._HOLE_SITE_1, object)
        # self.hole_site_2 = Site.get(self._mojo, self._HOLE_SITE_2, object)
        # self.hole_site_3 = Site.get(self._mojo, self._HOLE_SITE_3, object)
        # self.hole_site_4 = Site.get(self._mojo, self._HOLE_SITE_4, object)

        # self.sites  = [self.hole_site_1, self.hole_site_2, self.hole_site_3, self.hole_site_4]
        
        self.holes = [geom for geom in Body.get(self._mojo, self._HOLE, object).geoms]

class MontessoriPentagon(KinematicProp):
    """Montessori Pentagon."""
    _HOLE_SITE_1 = "hole_site_1"
    _HOLE_SITE_2 = "hole_site_2"
    _HOLE_SITE_3 = "hole_site_3"
    _HOLE_SITE_4 = "hole_site_4"
    _HOLE_SITE_5 = "hole_site_5"
    _HOLE = "hole_site"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/montessori/pentagon/pentagon.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
        # self.hole_site_1 = Site.get(self._mojo, self._HOLE_SITE_1, object)
        # self.hole_site_2 = Site.get(self._mojo, self._HOLE_SITE_2, object)
        # self.hole_site_3 = Site.get(self._mojo, self._HOLE_SITE_3, object)
        # self.hole_site_4 = Site.get(self._mojo, self._HOLE_SITE_4, object)
        # self.hole_site_5 = Site.get(self._mojo, self._HOLE_SITE_5, object)  
        # self.sites  = [self.hole_site_1, self.hole_site_2, self.hole_site_3, self.hole_site_4, self.hole_site_5]
        
        self.holes = [geom for geom in Body.get(self._mojo, self._HOLE, object).geoms]


class MontessoriRectangle(KinematicProp):
    """Montessori Rectangle."""
    _HOLE_SITE_1 = "hole_site_1"
    _HOLE_SITE_2 = "hole_site_2"
    
    _HOLE = "hole_site"
    
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/montessori/rectangle/rectangle.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
        # self.hole_site_1 = Site.get(self._mojo, self._HOLE_SITE_1, object)
        # self.hole_site_2 = Site.get(self._mojo, self._HOLE_SITE_2, object)
        # self.sites  = [self.hole_site_1, self.hole_site_2]
        
        self.holes = [geom for geom in Body.get(self._mojo, self._HOLE, object).geoms]



class MontessoriTriangle(KinematicProp):
    """Montessori Triangle."""
    _HOLE_SITE_1 = "hole_site_1"
    _HOLE_SITE_2 = "hole_site_2"
    _HOLE_SITE_3 = "hole_site_3"
    _HOLE = "hole_site"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/garage/montessori/triangle/triangle.xml"
    
    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
        # self.hole_site_1 = Site.get(self._mojo, self._HOLE_SITE_1, object)
        # self.hole_site_2 = Site.get(self._mojo, self._HOLE_SITE_2, object)
        # self.hole_site_3 = Site.get(self._mojo, self._HOLE_SITE_3, object)
        # self.sites  = [self.hole_site_1, self.hole_site_2, self.hole_site_3]
        
        self.holes = [geom for geom in Body.get(self._mojo, self._HOLE, object).geoms]


