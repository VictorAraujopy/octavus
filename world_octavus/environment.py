"""
Octavus environment in the Gymnasium format: the octopus has to move in a direction, on the sea floor.

Each episode a direction is drawn at random around the octopus and it has to go that way: there is no target to reach,
the episode just lasts 20 s. The brain sees the direction in its own body's frame (3 numbers, length 1), so it knows
which way to go however it is turned. The red ball only shows the direction in the viewer: it stays 1 m ahead that way.
Nothing changes by itself from one episode to the next: what the octopus has to learn is set by hand.
Stage 1 is crawling: with JET_ON = False the whole siphon ignores its commands (aim, tilt and jet), so the only way
to move is with the arms. Turn it on, by hand, when it crawls.

Each arm is a soft tentacle (see build_octopus.py): 16 segments in 4 sections, and each section has 4 muscles
(bend up/down, bend sideways, twist, stretch/shorten) plus the suckers of its segments.

Observation (305 numbers):
    obs[:288].reshape(8, 4, 9) -> per arm, per section: where its 4 muscles are (-1 to 1 of their reach),
                                  how fast they move, and how hard the section touches something
    obs[288:]                  -> body: direction (3), up (3), velocity (3), spin (3), height (1),
                                  siphon aim (1), siphon tilt (1), mantle water (1), jet stamina (1)

Action (163 numbers between -1 and 1), the arms first and the siphon last:
    action[:160].reshape(8, 4, 5) -> per arm, per section: bend_up, bend_side, twist, stretch, sucker
                                     (one sucker number drives the suckers of all the section's segments)
    action[160:163]               -> siphon: siphon_aim (funnel left/right), siphon_tilt (funnel up/down),
                                     jet (how hard it squirts)
    0 is always "no force" (relaxed muscle, loose sucker, no jet); suckers and jet treat anything below 0 as off

The jet squirts the water in the mantle: a full jet empties it in 0.44 s, and it refills in 0.4 s while the jet
rests (refilling only starts after 0.1 s relaxed in a row, so a squirt is a real squeeze, not a flicker).
The mantle works in a rhythm: once it runs empty it refills before squeezing again, so holding the jet on swims in pulses.
Jetting also stops the octopus's systemic heart, so it tires: 5 s of full jet in total, back after ~13 min of rest.

power is the metabolic cost in watts, what the food pays for: muscles pushing cost 4x their work (25% efficient),
muscles braking 1/1.2 of it, holding force up to 100 W per kg of muscle, growing with the square of the effort (even when nothing moves),
holding suckers nothing, the jet its hydrodynamic power at 25%, plus a common octopus's resting metabolism.

The reward is not decided here: the trainer passes a function reward_fn(info) -> float,
and the environment hands over the facts of each step in the info dict:
speed (m/s in the episode's direction, negative going the other way), flipped, action, previous_action, vertical_speed, height, spin, power (metabolic watts), holding_power (the part of power spent holding force, in any contraction), rigidity (0 to 1: how much each arm muscle has been holding the same command over the last ~second; 0 = relaxed or contracting and releasing), mass (kg), tips_touching (0 to 1: arms whose last section touches something), airborne (nothing touching the floor), facing (1 = eyes pointing the episode's direction, -1 = back to it), dt.

Watch the octopus moving randomly (on macOS the viewer needs mjpython):
    uv run mjpython world_octavus/environment.py
"""

import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from world_octavus.build_octopus import ARM_LENGTH, SECTIONS, SEGMENTS, section_of, segment_radius

XML = Path(__file__).resolve().parent / "octopus.xml"
JET_ON = False  # stage 1: crawl with the arms. True = the siphon works again (jet, aim and tilt)
MUSCLES = ("bend_up", "bend_side", "twist", "stretch")


class OctopusEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    physics_steps = 5
    # 20 s per episode: moving in a direction needs no time to arrive anywhere, and a training lap (2048 steps, ~51 s)
    # then sees ~2.5 different starts instead of not even one
    max_steps = 800
    marker_ahead = 1.0  # meters: the red ball shows the direction, always this far ahead of the octopus
    initial_noise = 0.1  # fraction of each joint's range
    # keep every observation number close to 1, so the brain's Tanh layers don't saturate
    muscle_speed_time = 0.1  # a muscle crossing its whole reach in 0.1 s reads 1
    touch_scale = 100.0  # a section gripping with its suckers presses up to ~90 N
    tip_touch = 0.005  # newtons: a resting tip lies on the floor with ~0.08 N, a tip in the water with 0
    spin_scale = 5.0
    # the mantle's jet cycle: 0.44 s squeezing, ~0.4 s refilling (measured in cuttlefish, Gladman & Askew 2022:
    # the octopus's own cycle wasn't found)
    jet_empty_time = 0.44
    jet_relaxed = 0.05
    # the mantle only starts drawing water after this long relaxed in a row (estimate); without it, flicking the
    # jet on and off every step gave a smooth nonstop jet. Delay + refill = the measured 0.4 s
    jet_refill_delay = 0.1
    jet_refill_time = 0.3
    # jetting stops the systemic heart, so it runs on an oxygen debt that only allows "a few metres" (Wells 1987):
    # ~5 s of full jet at its ~1 m/s top speed (our reading of "a few"). The debt (~22 ml O2/kg) is repaid at
    # ~100 ml O2/kg/h, the extra oxygen it can take up (2.4x routine, Wells 1983): ~13 min
    jet_stamina_time = 5.0
    jet_recovery_time = 790.0
    funnel_radius = 0.008  # the siphon capsule in octopus.xml (scaled with the head: the real funnel's width wasn't found)
    muscle_efficiency = 0.25
    braking_efficiency = 1.2
    # W per kg of muscle held at full force, even without moving. It grows with the square of the effort: light tone
    # uses the slow, thrifty fibres and a hard clench recruits the fast, costly ones (an estimate: no octopus value
    # found). At 10 W/kg in a straight line, holding every muscle at 75% cost ~0.07 per step and it stayed rigid
    holding_rate = 100.0
    # Hill's force-velocity: a muscle loses force the faster it shortens, and resists harder while being stretched.
    # Fastest shortening, in muscle lengths per second, measured in common octopus arms (Zullo et al. 2022):
    # longitudinal muscles 0.91 (bend, shorten, twist), transverse 0.36 (they squeeze the arm thinner to stretch it)
    longitudinal_vmax = 0.91
    transverse_vmax = 0.36
    hill_curvature = 0.25  # how sharply force drops with speed (Hill's a/F0): the classic value, an estimate
    eccentric_max = 1.5  # force while being stretched fast, times the held force (classic 1.5-1.8, an estimate)
    # rigidity: each arm muscle's command averaged over about this long. Holding the same force keeps the average high;
    # contracting and releasing (a stride, flailing) averages out near 0 (an estimate, chosen by eye)
    rigidity_time = 1.0

    def __init__(self, reward_fn=None, render_mode=None, xml=XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.viewer = None
        m = self.model

        self.torso = m.body("torso").id
        self.mass = m.body_subtreemass[self.torso]
        self.basal_power = self._resting_metabolism()
        self.n_arms = sum(m.body(i).name.startswith("arm") for i in range(m.nbody))
        motor = {m.actuator(i).name: i for i in range(m.nu)}
        tendon = lambda name: m.tendon(name).id
        sensor = lambda name: m.sensor_adr[m.sensor(name).id]

        # the brain speaks per section: 4 muscles, then one number for all the suckers of that section's segments
        drives = []
        for a in range(self.n_arms):
            for k in range(SECTIONS):
                drives += [[motor[f"sec{k}_{kind}{a}"]] for kind in MUSCLES]
                drives.append([motor[f"seg{i}_sucker{a}"] for i in range(SEGMENTS) if section_of(i) == k])
        drives += [[motor["siphon_aim"]], [motor["siphon_tilt"]], [motor["jet"]]]
        self.siphon_motors = [motor["siphon_aim"], motor["siphon_tilt"], motor["jet"]]
        self.action_of_motor = np.zeros(m.nu, dtype=int)
        for number, motors in enumerate(drives):
            self.action_of_motor[motors] = number
        self.n_actions = len(drives)

        self.section_muscles = np.array([[[tendon(f"sec{k}_{kind}{a}") for kind in MUSCLES]
                                          for k in range(SECTIONS)] for a in range(self.n_arms)])
        # a muscle's reach: how far its tendon goes when all its joints hit their limit
        joints = [m.wrap_objid[m.tendon_adr[t]:m.tendon_adr[t] + m.tendon_num[t]] for t in range(m.ntendon)]
        reach = np.array([np.abs(m.jnt_range[j]).max(axis=1).sum() for j in joints])
        self.muscle_reach = reach[self.section_muscles]
        # sums each segment's touch sensor into its section
        self.touch_to_section = np.zeros((self.n_arms * SECTIONS, m.nsensordata))
        for a in range(self.n_arms):
            for i in range(SEGMENTS):
                self.touch_to_section[a * SECTIONS + section_of(i), sensor(f"seg{i}_touch{a}")] = 1

        # every joint but the free one gets a start nudge proportional to its range (a slide only moves centimeters)
        joint_ids = np.arange(1, m.njnt)
        self.joint_qpos = m.jnt_qposadr[joint_ids]
        self.joint_noise = self.initial_noise * (m.jnt_range[joint_ids, 1] - m.jnt_range[joint_ids, 0]) / 2

        # the body as the XML writes it has straight arms poking ~15 cm into the floor, and that first shove made the
        # light arms blow up now and then: let it settle once with no muscle working, and start every episode there
        self.resting_pose = self._settle()
        self.resting_height = self.resting_pose[2]  # the head's center when it lies still on the floor

        # step() maps the brain's 0 to "no force", which needs every motor's range to include 0
        assert (m.actuator_ctrlrange[:, 0] <= 0).all() and (m.actuator_ctrlrange[:, 1] >= 0).all()
        self.muscles = np.isin(m.actuator_trntype, [mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_TENDON])
        # for the holding cost: how much muscle each motor has (a section's mass is shared by its 4 muscles)
        # and its full force, to turn the force it makes into an activation from 0 to 1
        self.muscle_mass = np.zeros(m.nu)
        for i in np.flatnonzero(self.muscles):
            target = m.actuator_trnid[i][0]
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
                bodies = np.unique(m.jnt_bodyid[joints[target]])
                self.muscle_mass[i] = m.body_mass[bodies].sum() / len(MUSCLES)
            else:
                self.muscle_mass[i] = m.body_mass[m.jnt_bodyid[target]]
        self.max_force = np.where(m.actuator_forcelimited.astype(bool),
                                  np.abs(m.actuator_forcerange).max(axis=1), np.abs(m.actuator_ctrlrange).max(axis=1))
        # for Hill: turn each arm muscle's tendon speed into the muscle's own lengths per second.
        # A bend/twist tendon is the sum of its section's joint angles: the muscle on that side, at the arm's radius,
        # shortens by radius x angle. A stretch tendon is the section's change in length, in meters
        self.arm_muscles = np.flatnonzero(m.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON)
        self.arm_muscle_tendons = m.actuator_trnid[self.arm_muscles, 0]
        self.strain_per_tendon = np.zeros(len(self.arm_muscles))
        self.is_stretch = np.zeros(len(self.arm_muscles), dtype=bool)
        for j, tendon in enumerate(self.arm_muscle_tendons):
            section, kind = m.tendon(tendon).name.rstrip("0123456789")[len("sec"):].split("_", 1)  # "sec2_bend_up3"
            segments = [i for i in range(SEGMENTS) if section_of(i) == int(section)]
            muscle_length = len(segments) * ARM_LENGTH / SEGMENTS
            if kind == "stretch":
                self.strain_per_tendon[j] = 1 / muscle_length
                self.is_stretch[j] = True
            else:
                self.strain_per_tendon[j] = np.mean([segment_radius(i) for i in segments]) / muscle_length
        self.commanded_activation = np.zeros(len(self.arm_muscles))
        self.held_command = np.zeros(len(self.arm_muscles))
        self.jet = m.actuator("jet").id
        self.max_thrust = m.actuator_gear[self.jet][2]
        self.siphon_qpos = m.jnt_qposadr[m.joint("siphon_aim").id]
        self.siphon_tilt_qpos = m.jnt_qposadr[m.joint("siphon_tilt").id]
        self.siphon_tilt_reach = m.jnt_range[m.joint("siphon_tilt").id][1]  # 45 degrees, in radians
        self.mantle_water = 1.0
        self.stamina = 1.0
        self.relaxed_time = 0.0
        self.refilling = False
        self.dt = m.opt.timestep * self.physics_steps
        self.direction = np.array([1.0, 0.0, 0.0])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n_actions,), np.float32)
        observation_size = self._observe().size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.resting_pose
        self.data.qpos[self.joint_qpos] += self.np_random.uniform(-1, 1, self.joint_noise.size) * self.joint_noise

        angle = self.np_random.uniform(0, 2 * np.pi)
        self.direction = np.array([np.cos(angle), np.sin(angle), 0.0])  # flat: rising or sinking doesn't count

        mujoco.mj_forward(self.model, self.data)
        self._place_marker()
        self.step_count = 0
        self.previous_action = np.zeros(self.n_actions)
        self.mantle_water = 1.0
        self.stamina = 1.0
        self.relaxed_time = 0.0
        self.refilling = False
        self.held_command[:] = 0.0
        return self._observe(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # 0 from the brain is always "no force": 0..1 scales up to the motor's top, 0..-1 down to its bottom.
        # Suckers and jet only go 0..1, so anything at or below 0 is off (before, 0 meant half suction and half jet)
        low, high = self.model.actuator_ctrlrange.T
        wanted = action[self.action_of_motor]
        self.data.ctrl[:] = np.where(wanted >= 0, wanted * high, wanted * -low)
        if not JET_ON:
            self.data.ctrl[self.siphon_motors] = 0.0  # stage 1: the siphon stays relaxed whatever the brain says
        # each arm muscle's command, -1 to 1, averaged over the last ~second (for the rigidity in the info dict)
        self.held_command += (wanted[self.arm_muscles] - self.held_command) * self.dt / self.rigidity_time
        position_before = self.data.xpos[self.torso].copy()
        self._squirt()
        commanded = self.data.ctrl[self.arm_muscles].copy()
        self.commanded_activation = np.abs(commanded) / self.max_force[self.arm_muscles]
        for _ in range(self.physics_steps):
            # Hill: the faster an arm muscle is shortening right now, the less of the commanded force it gets
            self.data.ctrl[self.arm_muscles] = commanded * self._force_velocity(commanded)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        moved = self.data.xpos[self.torso] - position_before
        self._place_marker()
        flipped = self.data.xmat[self.torso][8] < 0
        power, holding_power = self._metabolic_power()
        info = {
            "speed": moved @ self.direction / self.dt,
            "flipped": flipped,
            "action": action,
            "previous_action": self.previous_action,
            "vertical_speed": self.data.qvel[2],
            "height": self.data.xpos[self.torso][2],
            "spin": self.data.qvel[5],
            "power": power,
            "holding_power": holding_power,
            "rigidity": float(np.mean(self.held_command ** 2)),
            "mass": self.mass,
            "tips_touching": (self._section_touch()[:, -1] > self.tip_touch).mean(),
            "airborne": self.data.ncon == 0,
            "facing": self._facing(),
            "dt": self.dt,
        }
        reward = self.reward_fn(info) if self.reward_fn else 0.0
        self.previous_action = action
        # only time ends an episode. Flipping over doesn't: it has to right itself (ending it made dying early
        # a way out whenever living scored negative)
        terminated = False
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._observe(), reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.torso
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -35
        self.viewer.sync()
        time.sleep(self.dt)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _force_velocity(self, commanded):
        # each arm muscle's shortening speed, in its own lengths per second (negative = being stretched)
        speed = np.sign(commanded) * self.data.ten_velocity[self.arm_muscle_tendons] * self.strain_per_tendon
        vmax = np.where(self.is_stretch & (commanded > 0), self.transverse_vmax, self.longitudinal_vmax)
        shortening = np.clip(speed / vmax, 0.0, None)
        stretched = np.clip(-speed / vmax, 0.0, None)
        concentric = np.clip((1 - shortening) / (1 + shortening / self.hill_curvature), 0.0, None)
        # being stretched: the force rises toward eccentric_max, as steeply at 0 as it falls on the shortening side
        steepness = (1 + 1 / self.hill_curvature) / (self.eccentric_max - 1)
        eccentric = self.eccentric_max - (self.eccentric_max - 1) / (1 + steepness * stretched)
        return np.where(speed >= 0, concentric, eccentric)

    def _settle(self):
        # 3 s with every motor at 0 (no force): the arms fall onto the floor and the springs curl the tips
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=int(3 / self.model.opt.timestep))
        if self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise RuntimeError("the body blew up while settling into its resting pose")
        pose = self.data.qpos.copy()
        pose[:2] = 0.0  # back to the center of the floor
        return pose

    def _resting_metabolism(self):
        # watts a resting common octopus burns in 20 C water (Katsanevakis 2005: oxygen use from its weight),
        # at 13.39 J of food per mg of oxygen
        micromol_o2_per_hour = np.exp(25.24 - 6952.8 / 293.15) * (self.mass * 1000) ** 0.901
        mg_o2_per_hour = micromol_o2_per_hour * 0.032
        return mg_o2_per_hour * 13.39 / 3600

    def _metabolic_power(self):
        # arm and siphon muscles: work done pushing costs 1/0.25, work absorbed braking costs 1/1.2 (suckers hold for free)
        force = self.data.actuator_force
        work = (force * self.data.actuator_velocity)[self.muscles]
        muscles = work.clip(min=0).sum() / self.muscle_efficiency - work.clip(max=0).sum() / self.braking_efficiency
        # holding force costs even when nothing moves (without this, a muscle clenched at full force was free)
        activation = np.abs(force) / self.max_force
        # an arm muscle's effort is what was commanded: Hill lowers the force it gets while shortening, not the effort
        activation[self.arm_muscles] = self.commanded_activation
        activation = activation[self.muscles]
        holding = self.holding_rate * (activation ** 2 * self.muscle_mass[self.muscles]).sum()
        total = muscles + holding + self._jet_power() / self.muscle_efficiency + self.basal_power
        return total, holding

    def _squirt(self):
        # the mantle is a pump: it squirts while it has water and only refills once the jet has been relaxed
        # for a moment. Jetting also stops the heart, so a stamina runs out and only comes back while resting
        requested = self.data.ctrl[self.jet]
        drain = self.dt / self.jet_empty_time
        tire = self.dt / self.jet_stamina_time
        # once empty, the mantle fills up before the next squeeze even if the jet is still wanted: holding the jet
        # on swims in pulses. Without this, a brain that held it on emptied the mantle once and never jetted again
        if self.mantle_water <= 0.0:
            self.refilling = True
        elif self.mantle_water >= 1.0:
            self.refilling = False
        squeezing = requested >= self.jet_relaxed and not self.refilling
        jet = min(requested, self.mantle_water / drain, self.stamina / tire) if squeezing else 0.0
        self.data.ctrl[self.jet] = jet
        # max(0, ...): float rounding would leave them at -1e-16
        self.mantle_water = max(0.0, self.mantle_water - jet * drain)
        self.stamina = max(0.0, self.stamina - jet * tire)
        if squeezing:
            self.relaxed_time = 0.0
        else:
            self.relaxed_time += self.dt
        if self.relaxed_time >= self.jet_refill_delay:
            self.mantle_water = min(1.0, self.mantle_water + self.dt / self.jet_refill_time)
            self.stamina = min(1.0, self.stamina + self.dt / self.jet_recovery_time)

    def _jet_power(self):
        # pushing water out: thrust T needs the water leaving the funnel at u = sqrt(T / (density * area)), costing T * u / 2
        thrust = self.data.ctrl[self.jet] * self.max_thrust
        area = np.pi * self.funnel_radius ** 2
        return thrust * np.sqrt(thrust / (self.model.opt.density * area)) / 2

    def _section_touch(self):
        return (self.touch_to_section @ self.data.sensordata).reshape(self.n_arms, SECTIONS)

    def _facing(self):
        # cosine of the angle between where the eyes point (the body's +x) and the episode's direction, on the floor plane
        rotation = self.data.xmat[self.torso].reshape(3, 3)
        seen = rotation.T @ self.direction
        return seen[0] / (np.hypot(seen[0], seen[1]) + 1e-8)

    def _place_marker(self):
        # the red ball, only for the eyes: on the floor, marker_ahead in front of the octopus along the direction
        ahead = self.data.xpos[self.torso] + self.marker_ahead * self.direction
        self.data.mocap_pos[0] = [ahead[0], ahead[1], self.resting_height]

    def _observe(self):
        rotation = self.data.xmat[self.torso].reshape(3, 3)
        position = self.data.xpos[self.torso]

        per_section = np.concatenate([
            self.data.ten_length[self.section_muscles] / self.muscle_reach,
            self.data.ten_velocity[self.section_muscles] * self.muscle_speed_time / self.muscle_reach,
            self._section_touch()[..., None] / self.touch_scale,
        ], axis=2)

        direction_seen = rotation.T @ self.direction
        up_seen = rotation.T @ [0.0, 0.0, 1.0]
        torso_velocity = rotation.T @ self.data.qvel[:3]
        torso_spin = self.data.qvel[3:6] / self.spin_scale
        height = [position[2]]
        siphon = [self.data.qpos[self.siphon_qpos] / np.pi,
                  self.data.qpos[self.siphon_tilt_qpos] / self.siphon_tilt_reach,
                  self.mantle_water,
                  self.stamina]

        return np.concatenate([
            per_section.ravel(),
            direction_seen,
            up_seen,
            torso_velocity,
            torso_spin,
            height,
            siphon,
        ]).astype(np.float32)


if __name__ == "__main__":
    env = OctopusEnv(render_mode="human")
    observation, _ = env.reset(seed=0)
    print(f"the brain receives {observation.size} numbers and returns {env.action_space.shape[0]}")

    speeds = []
    while env.viewer is None or env.viewer.is_running():
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        speeds.append(info["speed"])
        if terminated or truncated:
            print(f"episode over: {np.mean(speeds):+.3f} m/s in the direction, flipped={info['flipped']}")
            speeds = []
            env.reset()
    env.close()
