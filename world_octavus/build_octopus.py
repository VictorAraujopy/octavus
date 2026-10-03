"""
Builds octopus.xml: Octavus's body and the sea floor, in MuJoCo's MJCF format.
Edit this file, not the XML, then run:  uv run python world_octavus/build_octopus.py

Each arm is a soft tentacle: SEGMENTS pieces tapering to the tip. Every segment can bend (up/down and sideways),
twist and stretch, held by springs, so the arm sways in the water on its own. Like the octopus's muscle bands,
a muscle doesn't move one joint: it pulls a whole section of the arm (a fixed tendon over that section's joints).
Per section, 4 muscles (bend up/down, bend sideways, twist, stretch/shorten) and the suckers of its segments.

Real numbers used (common octopus, Octopus vulgaris): an arm stretches up to 70%, shortens ~20% and pulls with
~40 N (Margheri 2012); it is 10-16x longer than its base is wide (Kazakidi 2015) and ~12x thinner at the tip
(Kier & Stella 2007); arms are ~4.5x the mantle (Mazzolai 2013); its tissue is 1.05-1.07 g/cm3 (Rahman &
Driscoll 1994). Twist and how far one segment bends are estimates (no measured value found).
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

ARMS = 8
SEGMENTS = 16
SECTIONS = 4
ARM_LENGTH = 0.6
# 3.75 cm across at the base: length / width 16, the thinnest measured. Wider ones (down to 10) would make
# the arms alone heavier than a whole octopus of this size (0.6-2 kg)
BASE_RADIUS = 0.01875
TIP_RADIUS = 0.002  # ~4 mm across at the last segment
TISSUE_DENSITY = 1060.0  # kg/m3
BEND = 45  # degrees per segment, both ways: 16 segments can curl the tip into a spiral
TWIST = 15  # degrees per segment
SHORTEN, STRETCH = 0.2, 0.7  # fraction of the segment's length
PULL = 40.0  # newtons, the longitudinal muscles at the base; thinner sections pull less
TWIST_SHARE = 0.25  # oblique muscles are a thin layer: a quarter of the bending strength
GRIP = 120.0  # newtons of sucker adhesion per arm, spread by sucker size
BUOYANCY = 1025.0 / TISSUE_DENSITY  # 0.967: the water holds this much of its weight, so it weighs 3.3% in water
# the head carries the mantle, a sac full of sea water (which weighs nothing in water), so it is almost neutral:
# only the organs pull it down a little (estimate). With the arms' buoyancy, lying sideways it nosedived
MANTLE_BUOYANCY = 0.99
DAMPING = 0.05  # seconds: each joint's damping (the tissue's viscosity) is its muscle stiffness times this, so it doesn't ring
# a relaxed octopus arm is floppy: its longitudinal tissue resists stretch with ~10-40 kPa (O. rubescens, JEB 2024)
# against ~107 kPa of active muscle stress (Zullo et al. 2022), so bending a segment 45 degrees relaxed takes ~3-11%
# of the muscle's full torque. The springs used to take 100%: a relaxed arm couldn't fold back in the water and it
# held its arms tucked with muscle to swim
SPRING_SHARE = 0.07
# MuJoCo's suckers (adhesion) only pull on contacts, and a resting arm hovers ~1 mm above the floor, so a sucker
# turned on often had nothing to grip. A real sucker protrudes and moulds onto the floor: let it reach 5 mm
# (margin), without that margin pushing anything away (gap). An estimate: 10 mm made the grip flicker again
SUCKER_REACH = 0.005
ARM_ROOT = 0.04  # meters from the mouth to where each arm starts
ARM_SLOPE = 25  # degrees the arm's base slopes down, so the arm reaches the floor
# the web joins the arms out to ~25% of the arm's length from the mouth (web depth index 22-29 in O. vulgaris,
# Leite et al. 2008). Only drawn: it has no weight and pushes no water
WEB_DEPTH = 0.25 * ARM_LENGTH
# a resting arm isn't straight: from segment CURL_FROM on, its springs rest bent sideways, more and more toward
# the tip (up to CURL degrees per segment), so the tip curls into a loose spiral. A pose, not a measured number
CURL = 30
CURL_FROM = 6

XML = Path(__file__).resolve().parent / "octopus.xml"


def segment_radius(i):
    return BASE_RADIUS + (TIP_RADIUS - BASE_RADIUS) * i / (SEGMENTS - 1)


def section_of(i):
    return i * SECTIONS // SEGMENTS


def joint(name, kind, axis, low, high, muscle_stiffness, rest=0.0):
    # muscle_stiffness: the spring the muscle's full force would take to its limit. The relaxed tissue is only
    # SPRING_SHARE of it (a floppy arm); its damping (the tissue's viscosity) stays at the muscle's scale
    # rest: where the spring pulls the joint back to when no muscle works (springref)
    kind_attr = ' type="slide"' if kind == "slide" else ""
    return (f'<joint name="{name}"{kind_attr} axis="{axis}" range="{low:.4g} {high:.4g}" springref="{rest:.4g}" '
            f'stiffness="{SPRING_SHARE * muscle_stiffness:.4g}" damping="{muscle_stiffness * DAMPING:.4g}"/>')


def arm():
    length = ARM_LENGTH / SEGMENTS
    torque = lambda i: PULL * segment_radius(i)  # muscle force x lever arm (the arm's radius)
    grips = np.array([segment_radius(i) for i in range(SEGMENTS)])
    grips = GRIP * grips / grips.sum()

    xml, close = "", ""
    for i in range(SEGMENTS):
        r = segment_radius(i)
        pos = "0 0 0" if i == 0 else f"{length:.4f} 0 0"
        tilt = f' euler="0 {ARM_SLOPE} 0"' if i == 0 else ""
        material = "skin" if section_of(i) < 2 else "tip"
        curl = CURL * max(0, i - CURL_FROM) / (SEGMENTS - 1 - CURL_FROM)
        bend_k = torque(i) / np.radians(BEND)
        twist_k = TWIST_SHARE * torque(i) / np.radians(TWIST)
        stretch_k = PULL * r / BASE_RADIUS / (STRETCH * length)
        # weighed as the cylinder it stands for: MuJoCo would weigh the whole capsule, and the round ends of
        # neighbouring segments overlap, which made the arms 2.4x too heavy
        mass = TISSUE_DENSITY * np.pi * r ** 2 * length
        xml += f"""
<body name="seg{i}_" pos="{pos}"{tilt} gravcomp="{BUOYANCY}">
  {joint(f"seg{i}_stretch", "slide", "1 0 0", -SHORTEN * length, STRETCH * length, stretch_k)}
  {joint(f"seg{i}_bend_up", "hinge", "0 1 0", -BEND, BEND, bend_k)}
  {joint(f"seg{i}_bend_side", "hinge", "0 0 1", -BEND, BEND, bend_k, rest=curl)}
  {joint(f"seg{i}_twist", "hinge", "1 0 0", -TWIST, TWIST, twist_k)}
  <geom material="{material}" type="capsule" fromto="0 0 0 {length:.4f} 0 0" size="{r:.5f}" mass="{mass:.5g}" margin="{SUCKER_REACH}" gap="{SUCKER_REACH}"/>
  <site name="seg{i}_start" size="0.001" rgba="0 0 0 0"/>
  <site name="seg{i}_end" pos="{length:.4f} 0 0" size="0.001" rgba="0 0 0 0"/>
  <geom class="visual" material="sucker" type="sphere" size="{0.4 * r:.4f}" pos="{length / 2:.4f} 0 {-0.85 * r:.4f}"/>
  <site name="seg{i}_touch" type="capsule" fromto="0 0 0 {length:.4f} 0 0" size="{1.2 * r:.4f}" rgba="0 0 0 0"/>"""
        close += "</body>\n"

    tendons, muscles, suckers, sensors = "", "", "", ""
    # stretching slides a segment away from the one before it and opens a gap. A drawn-only tube from the end of one
    # segment to the start of the next fills it however the joint bends or stretches. It used to be a straight capsule
    # fixed to the segment: bent, it poked out of the side of the segment before (the arm looked like loose sticks).
    # A tendon with no stiffness, damping, limit or motor pulls on nothing
    for i in range(SEGMENTS):
        before = "arm_root" if i == 0 else f"seg{i - 1}_end"
        material = "skin" if section_of(i) < 2 else "tip"
        tendons += (f'<spatial name="seg{i}_gap" width="{0.97 * segment_radius(i):.5f}" material="{material}">'
                    f'<site site="{before}"/><site site="seg{i}_start"/></spatial>\n')
    for k in range(SECTIONS):
        segs = [i for i in range(SEGMENTS) if section_of(i) == k]
        lever = np.mean([segment_radius(i) for i in segs])
        strength = {"bend_up": PULL * lever, "bend_side": PULL * lever,
                    "twist": TWIST_SHARE * PULL * lever, "stretch": PULL * lever / BASE_RADIUS}
        for kind, gear in strength.items():
            joints = "".join(f'<joint joint="seg{i}_{kind}" coef="1"/>' for i in segs)
            tendons += f'<fixed name="sec{k}_{kind}">{joints}</fixed>\n'
            # shortening only gets the force its spring holds at -20%, or it would push past the limit
            low = -SHORTEN / STRETCH if kind == "stretch" else -1
            muscles += f'<motor name="sec{k}_{kind}" tendon="sec{k}_{kind}" ctrlrange="{low:.3g} 1" gear="{gear:.4g}"/>\n'
    for i in range(SEGMENTS):
        suckers += f'<adhesion name="seg{i}_sucker" body="seg{i}_" ctrlrange="0 1" gain="{grips[i]:.4g}"/>\n'
        sensors += f'<touch name="seg{i}_touch" site="seg{i}_touch"/>\n'
    return xml + close, tendons, muscles + suckers, sensors


def octopus(skin=""):
    arm_bodies, tendons, actuators, sensors = arm()
    return f"""<!--
  Generated by build_octopus.py: edit that file, not this one.
  Octavus's body and the sea floor. {ARMS} soft arms of {SEGMENTS} segments in {SECTIONS} sections, and a siphon.
  View it (from the project root; the path must be absolute, hence $PWD):
    uv run python -m mujoco.viewer --mjcf="$PWD/world_octavus/octopus.xml"
-->
<mujoco model="octavus">
  <compiler angle="degree" autolimits="true"/>

  <!-- sea water: Earth's gravity, and each body's gravcomp is the buoyancy (MuJoCo's fluid adds drag, lift and the water dragged along, not buoyancy).
       Euler + sparse: the soft arms have ~500 joints, and this integrator is ~7x faster for a body shaped like a tree -->
  <option timestep="0.005" integrator="Euler" jacobian="sparse" gravity="0 0 -9.81" density="1025" viscosity="0.001"/>

  <visual>
    <rgba haze="0.05 0.25 0.35 1" fog="0.03 0.18 0.26 1"/>
    <map fogstart="1.5" fogend="7"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="water" type="skybox" builtin="gradient" rgb1="0.1 0.4 0.55" rgb2="0.01 0.06 0.1" width="512" height="512"/>
    <texture name="sand" type="2d" builtin="checker" rgb1="0.78 0.71 0.52" rgb2="0.72 0.65 0.47" width="512" height="512"/>
    <material name="sand" texture="sand" texrepeat="12 12"/>
    <!-- skin with darker specks; cube textures wrap around round shapes -->
    <texture name="skin" type="cube" builtin="flat" rgb1="0.8 0.32 0.24" rgb2="0.8 0.32 0.24" mark="random" markrgb="0.55 0.18 0.13" random="0.06" width="256" height="256"/>
    <material name="skin" texture="skin" rgba="1 1 1 1" specular="0.3" shininess="0.4"/>
    <texture name="tip" type="cube" builtin="flat" rgb1="0.88 0.42 0.32" rgb2="0.88 0.42 0.32" mark="random" markrgb="0.62 0.25 0.18" random="0.05" width="256" height="256"/>
    <material name="tip" texture="tip" rgba="1 1 1 1" specular="0.3" shininess="0.4"/>
    <material name="sucker" rgba="1 0.8 0.72 1"/>
    <!-- the web: the skin's color and shine without its texture (a drawn skin has no texture coordinates) -->
    <material name="web" rgba="0.8 0.32 0.24 1" specular="0.3" shininess="0.4"/>
    <material name="eye" rgba="0.86 0.68 0.3 1" specular="0.8" shininess="0.9"/>
    <material name="pupil" rgba="0.05 0.05 0.05 1"/>
  </asset>

  <default>
    <!-- slippery skin (mucus on wet sand): the suckers do the gripping. The floor inherits it: MuJoCo uses the larger of the two.
         fluidshape ellipsoid: water forces for long thin shapes (drag sideways vs lengthwise, lift, the water it drags along) -->
    <geom material="skin" contype="2" conaffinity="1" friction="0.3 0.02 0.01" fluidshape="ellipsoid" density="{TISSUE_DENSITY:g}"/>
    <!-- armature: a little made-up inertia per joint, without it the thin joints blow up. 0.002 never blew up in 52
         episodes of random flailing (0.001 did in 2 of 17); 0.01 made the 4 mm tip move like lead (4 deg in 0.1 s, now 19) -->
    <joint armature="0.002"/>
    <default class="visual">
      <geom contype="0" conaffinity="0" density="0" fluidshape="none"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.55 0.7 0.8" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="sand" contype="1" conaffinity="1" fluidshape="none"/>

    <!-- a ball that only shows the episode's direction: environment.py keeps it 1 m ahead of the octopus -->
    <body name="target" mocap="true" pos="2 0 0.135">
      <geom class="visual" type="sphere" size="0.12" rgba="1 0.25 0.2 0.7"/>
    </body>

    <body name="torso" pos="0 0 0.12" gravcomp="{MANTLE_BUOYANCY}">
      <freejoint name="root"/>
      <camera name="follow" mode="trackcom" pos="0 -1.8 1.1" xyaxes="1 0 0 0 0.53 0.848"/>
      <!-- the mantle is the head's physical part: its weight (with the water inside) and the water it pushes.
           13 cm long: the arms are ~4.5x the mantle (Mazzolai 2013) -->
      <geom name="mantle" type="ellipsoid" size="0.052 0.0492 0.066" pos="-0.014 0 0.022" euler="0 -28 0"/>
      <!-- the head under the mantle, where the eyes are: only drawn, so its weight isn't counted twice -->
      <geom name="head" class="visual" type="sphere" size="0.048"/>
      <!-- amber eyes half sunk into the sides of the head, with the octopus's horizontal slit pupil -->
      <geom class="visual" material="eye" type="sphere" size="0.0136" pos="0.004 0.04 0.02"/>
      <geom class="visual" material="eye" type="sphere" size="0.0136" pos="0.004 -0.04 0.02"/>
      <geom class="visual" material="pupil" type="ellipsoid" size="0.0076 0.0012 0.0024" pos="0.0052 0.05176 0.02588" euler="26.6 0 0"/>
      <geom class="visual" material="pupil" type="ellipsoid" size="0.0076 0.0012 0.0024" pos="0.0052 -0.05176 0.02588" euler="-26.6 0 0"/>

      <replicate count="{ARMS}" euler="0 0 {360 / ARMS:g}">
        <body name="arm" pos="{ARM_ROOT * np.cos(np.radians(180 / ARMS)):.5f} {ARM_ROOT * np.sin(np.radians(180 / ARMS)):.5f} -0.02" euler="0 0 {180 / ARMS:g}">
          <site name="arm_root" size="0.001" rgba="0 0 0 0"/>
{arm_bodies}
        </body>
      </replicate>

      <!-- the siphon (funnel): swivels around the head (aim) and tips up or down (tilt), and squirts the mantle's water.
           At rest it points 20 degrees down; the tilt adds up to 45 degrees either way, so like a real octopus it can
           steer the jet up or down (a rudder to keep the head from nosediving). No collision: it only aims the jet -->
      <body name="siphon" pos="0 0 -0.012" gravcomp="{BUOYANCY}">
        <joint name="siphon_aim" axis="0 0 1" range="-180 180" damping="0.5"/>
        <joint name="siphon_tilt" axis="0 1 0" range="-45 45" damping="0.5"/>
        <!-- the siphon's mass (what the aim and tilt muscles move), invisible -->
        <geom type="capsule" fromto="0.032 0 0 0.064 0 -0.0116" size="0.008" contype="0" conaffinity="0" fluidshape="none" rgba="0 0 0 0"/>
        <!-- what shows: just the funnel's mouth, barely out from under the head -->
        <geom class="visual" material="skin" type="capsule" fromto="0.038 0 0 0.0432 0 -0.002" size="0.0036"/>
        <!-- the jet pushes the body opposite to where the water leaves: backwards and a bit up -->
        <site name="jet" zaxis="-0.94 0 0.342" size="0.004" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
{skin}

  <tendon>
{tendons}  </tendon>

  <actuator>
{actuators}    <position name="siphon_aim" joint="siphon_aim" kp="8" kv="0.3" forcerange="-4 4" inheritrange="1"/>
    <position name="siphon_tilt" joint="siphon_tilt" kp="8" kv="0.3" forcerange="-4 4" inheritrange="1"/>
    <!-- up to 32 N, set by how it swims: mantle first with the arms relaxed, like a common octopus (Huffard 2006), its top
         speed is ~1 m/s, the measured one (Wells, via Huffard 2006). It was 40 N with stiff arm springs; floppy arms drag less.
         The environment stops it when the mantle is empty -->
    <motor name="jet" site="jet" ctrlrange="0 1" gear="0 0 32 0 0 0"/>
  </actuator>

  <sensor>
{sensors}  </sensor>
</mujoco>
"""


def web_skin(body_xml):
    """The web between the arms as one drawn sheet (a MuJoCo skin) whose points ride on two arms at once."""
    # the body without the web, in its starting pose (every joint at 0): where each arm segment is
    model = mujoco.MjModel.from_xml_string(body_xml)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    length = ARM_LENGTH / SEGMENTS
    middle_depth = (WEB_DEPTH - ARM_ROOT) / np.cos(np.radians(ARM_SLOPE))  # along the arm, halfway between two arms
    rows, columns = 6, 8  # points out from the arm roots, and across from one arm to the next

    def point_on_arm(arm, s):
        # the point s meters along the arm's centerline, and the segment holding it
        i = min(int(s / length), SEGMENTS - 1)
        segment = model.body(f"seg{i}_{arm}").id
        return data.xpos[segment] + data.xmat[segment].reshape(3, 3) @ [s - i * length, 0, 0], segment

    vertices, faces = [], []
    bones = {}  # segment -> the points it carries and how much of each
    for arm in range(ARMS):
        neighbour = (arm + 1) % ARMS
        first = len(vertices)
        for row in range(rows + 1):
            for column in range(columns + 1):
                share = column / columns  # 0 = on this arm, 1 = on the neighbour
                # the edge sags toward the mouth between two arms and runs 30% further along each arm (drawing choice)
                depth = middle_depth * (1 + 0.3 * (1 - np.sin(np.pi * share)))
                here, here_segment = point_on_arm(arm, depth * row / rows)
                there, there_segment = point_on_arm(neighbour, depth * row / rows)
                vertex = len(vertices)
                vertices.append((1 - share) * here + share * there)
                # a point halfway between two arms follows each of them by half: it stretches when they part
                if share < 1:
                    bones.setdefault(here_segment, []).append((vertex, 1 - share))
                if share > 0:
                    bones.setdefault(there_segment, []).append((vertex, share))
        for row in range(rows):
            for column in range(columns):
                corner = first + row * (columns + 1) + column
                faces.append((corner, corner + columns + 1, corner + 1))
                faces.append((corner + columns + 1, corner + columns + 2, corner + 1))

    # MuJoCo draws a skin from one side only: a second copy facing down makes it show from below too
    underside = len(vertices)
    vertices += vertices
    faces += [(c + underside, b + underside, a + underside) for a, b, c in faces]
    for carried in bones.values():
        carried += [(vertex + underside, weight) for vertex, weight in carried]

    bone_xml = ""
    for segment, carried in bones.items():
        position = " ".join(f"{x:.5f}" for x in data.xpos[segment])
        rotation = " ".join(f"{x:.5f}" for x in data.xquat[segment])
        ids = " ".join(str(vertex) for vertex, _ in carried)
        weights = " ".join(f"{weight:.4f}" for _, weight in carried)
        bone_xml += (f'      <bone body="{model.body(segment).name}" bindpos="{position}" bindquat="{rotation}" '
                     f'vertid="{ids}" vertweight="{weights}"/>\n')
    vertex_xml = " ".join(f"{x:.5f}" for point in vertices for x in point)
    face_xml = " ".join(str(index) for face in faces for index in face)
    return (f'  <deformable>\n    <skin name="web" material="web" vertex="{vertex_xml}" face="{face_xml}">\n'
            f'{bone_xml}    </skin>\n  </deformable>')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=XML)
    out = parser.parse_args().out
    body = octopus()
    out.write_text(octopus(skin=web_skin(body)))
    print(f"wrote {out}")
