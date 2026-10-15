"""
Octavus in the water, in the Gymnasium format: the body, its muscles and its senses, with no task yet.
A task builds on this and adds its own goal and reward.

Each arm is a soft tentacle (see build_octopus.py): 16 segments in 4 sections, and each section has 4 muscles
(bend up/down, bend sideways, twist, stretch/shorten) plus the suckers of its segments.

Observation (302 numbers):
    obs[:288].reshape(8, 4, 9) -> per arm, per section: where its 4 muscles are (-1 to 1 of their reach),
                                  how fast they move, and how hard the section touches something
    obs[288:]                  -> body: up (3), velocity (3), spin (3), height (1),
                                  siphon aim (1), siphon tilt (1), mantle water (1), jet stamina (1)

Action (163 numbers between -1 and 1), the arms first and the siphon last:
    action[:160].reshape(8, 4, 5) -> per arm, per section: bend_up, bend_side, twist, stretch, sucker
                                     (one sucker number drives all the suckers of the section)
    action[160:163]               -> siphon: siphon_aim (left/right), siphon_tilt (up/down), jet (how hard it squirts)
    An arm muscle's number is a posture: 0 is limp, and further out it heads for that share of the way from the
    resting pose to the joints' limit (-1 / 1), held firmer the further out (the arm's reflex, see posture_gain).
    Suckers and jet go 0 to 1; at or below 0 they are off.

The jet squirts the mantle's water in pulses (it empties in 0.44 s and refills in 0.4 s) and it tires:
5 s of full jet in total, back after ~13 min of rest.

info["power"] is the metabolic cost in watts: muscle work at 25% efficiency (braking 1/1.2), holding force,
the jet's hydrodynamic power at 25%, and a resting common octopus's metabolism.
info["flipped"] is True while the octopus lies upside down.

Watch it flail at random (on macOS the viewer needs mjpython):
    uv run mjpython world_octavus/environment.py
"""

import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from world_octavus.build_octopus import ARM_LENGTH, SECTIONS, SEGMENTS, XML, section_of, segment_radius

MUSCLES = ("bend_up", "bend_side", "twist", "stretch")


class OctopusEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    physics_steps = 5  # 5 x 5 ms: the brain acts every 0.025 s
    max_steps = 800  # 20 s per episode
    initial_noise = 0.1  # each joint starts nudged by up to this share of its range
    # keep every observation number close to 1, so the brain's layers don't saturate
    muscle_speed_time = 0.1  # a muscle crossing its whole reach in 0.1 s reads 1
    touch_scale = 100.0  # a section gripping with its suckers presses up to ~90 N
    spin_scale = 5.0
    # the mantle's jet cycle, measured in cuttlefish (Gladman & Askew 2022: no octopus value found)
    jet_empty_time = 0.44
    jet_relaxed = 0.05  # a jet command below this counts as relaxed
    jet_refill_delay = 0.1  # relaxed this long in a row before it refills (estimate): a squirt is a real squeeze
    jet_refill_time = 0.3
    # jetting stops the systemic heart: the oxygen debt allows "a few metres" (Wells 1987), ~5 s of full jet,
    # repaid in ~13 min (Wells 1983)
    jet_stamina_time = 5.0
    jet_recovery_time = 790.0
    funnel_radius = 0.008  # the siphon capsule in octopus.xml
    muscle_efficiency = 0.25
    braking_efficiency = 1.2
    # W per kg of muscle held at full force, growing with the square of the effort (estimate: no octopus value found)
    holding_rate = 100.0
    # Hill's force-velocity: fastest shortening in muscle lengths per second, common octopus arms (Zullo et al. 2022)
    longitudinal_vmax = 0.91  # bend, shorten, twist
    transverse_vmax = 0.36  # stretch (the arm squeezed thinner)
    hill_curvature = 0.25  # Hill's a/F0, the classic value (estimate)
    eccentric_max = 1.5  # force while being stretched fast, times the held force (classic 1.5-1.8, estimate)
    # the stretch muscles lose force over the last 15% of their range, so the soft joint limit holds (width: estimate)
    stretch_fade = 0.15
    # the arm's reflex (2/3 of an octopus's neurons are in its arms: the brain says where, the arm works out how):
    # push toward the asked posture with this share of full force per whole reach, times the command's firmness
    posture_gain = 1.0

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
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

        # the XML's straight arms poke into the floor, and that first shove can blow the light arms up:
        # settle once with no muscle working, and start every episode from there
        self.resting_pose = self._settle()

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
        self.tendon_reach = reach[self.arm_muscle_tendons]  # a stretch tendon's reach is its section's full +70%
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
        # posture 0 is each section as it lies settled on the floor, and -1 / 1 its joints' limits added up
        mujoco.mj_resetData(m, self.data)
        self.data.qpos[:] = self.resting_pose
        mujoco.mj_forward(m, self.data)
        self.rest_length = self.data.ten_length[self.arm_muscle_tendons].copy()
        arm_joints = [joints[t] for t in self.arm_muscle_tendons]
        self.shortest = np.array([m.jnt_range[j, 0].sum() for j in arm_joints])
        self.longest = np.array([m.jnt_range[j, 1].sum() for j in arm_joints])
        self.arm_force_low, self.arm_force_high = m.actuator_ctrlrange[self.arm_muscles].T
        self.posture = self.rest_length.copy()
        self.firmness = np.zeros(len(self.arm_muscles))
        self.commanded_activation = np.zeros(len(self.arm_muscles))
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

        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n_actions,), np.float32)
        observation_size = self._observe().size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.resting_pose
        self.data.qpos[self.joint_qpos] += self.np_random.uniform(-1, 1, self.joint_noise.size) * self.joint_noise
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.mantle_water = 1.0
        self.stamina = 1.0
        self.relaxed_time = 0.0
        self.refilling = False
        return self._observe(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # suckers and siphon: 0..1 scales up to the motor's top, 0..-1 down to its bottom. Arm muscles are postures, below
        low, high = self.model.actuator_ctrlrange.T
        wanted = action[self.action_of_motor]
        self.data.ctrl[:] = np.where(wanted >= 0, wanted * high, wanted * -low)
        self.posture = self._posture_target(wanted[self.arm_muscles])
        self.firmness = np.abs(wanted[self.arm_muscles])  # 0 = limp, 1 = the full reflex
        self._squirt()
        force_sum = np.zeros(len(self.arm_muscles))
        for _ in range(self.physics_steps):
            reflex = self._reflex()
            # Hill: the faster an arm muscle is shortening right now, the less of the reflex's force it gets
            self.data.ctrl[self.arm_muscles] = reflex * self._force_velocity(reflex) * self._force_length(reflex)
            force_sum += np.abs(reflex)
            mujoco.mj_step(self.model, self.data)
        # the effort is what the reflex asked for: Hill lowers the force it gets while shortening, not the effort
        self.commanded_activation = force_sum / self.physics_steps / self.max_force[self.arm_muscles]
        self.step_count += 1

        info = {
            "power": self._metabolic_power(),
            "flipped": self.data.xmat[self.torso][8] < 0,  # the head's up axis points down
        }
        # only time ends an episode: flipping over doesn't, it has to right itself
        terminated = False
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._observe(), 0.0, terminated, truncated, info

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

    def _posture_target(self, command):
        # where each section should go: 0..1 from the resting pose out to one limit, 0..-1 out to the other
        toward_longest = self.rest_length + command * (self.longest - self.rest_length)
        toward_shortest = self.rest_length + command * (self.rest_length - self.shortest)
        return np.where(command >= 0, toward_longest, toward_shortest)

    def _reflex(self):
        # push toward the posture, harder the further away and the firmer the command, within what each muscle can pull
        distance = self.posture - self.data.ten_length[self.arm_muscle_tendons]
        force = self.posture_gain * self.firmness * distance / self.tendon_reach
        return np.clip(force, self.arm_force_low, self.arm_force_high)

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

    def _force_length(self, commanded):
        # a stretch muscle pushing outward fades over the last stretch_fade of its reach; every other muscle keeps its force
        stretched = self.data.ten_length[self.arm_muscle_tendons] / self.tendon_reach  # 1 = the section at its full +70%
        fade = np.clip((1 - stretched) / self.stretch_fade, 0.0, 1.0)
        return np.where(self.is_stretch & (commanded > 0), fade, 1.0)

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
        # holding force costs even when nothing moves
        activation = np.abs(force) / self.max_force
        activation[self.arm_muscles] = self.commanded_activation
        activation = activation[self.muscles]
        holding = self.holding_rate * (activation ** 2 * self.muscle_mass[self.muscles]).sum()
        return muscles + holding + self._jet_power() / self.muscle_efficiency + self.basal_power

    def _squirt(self):
        # the mantle is a pump: it squirts while it has water and only refills once the jet has been relaxed
        # for a moment. Jetting also stops the heart, so a stamina runs out and only comes back while resting
        requested = self.data.ctrl[self.jet]
        drain = self.dt / self.jet_empty_time
        tire = self.dt / self.jet_stamina_time
        # once empty, the mantle fills up before the next squeeze even if the jet is still wanted: held on, it pulses
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

    def _observe(self):
        rotation = self.data.xmat[self.torso].reshape(3, 3)
        position = self.data.xpos[self.torso]

        per_section = np.concatenate([
            self.data.ten_length[self.section_muscles] / self.muscle_reach,
            self.data.ten_velocity[self.section_muscles] * self.muscle_speed_time / self.muscle_reach,
            self._section_touch()[..., None] / self.touch_scale,
        ], axis=2)

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

    powers = []
    while env.viewer is None or env.viewer.is_running():
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        powers.append(info["power"])
        if terminated or truncated:
            print(f"episode over: {np.mean(powers):.0f} W on average, flipped={info['flipped']}")
            powers = []
            env.reset()
    env.close()
