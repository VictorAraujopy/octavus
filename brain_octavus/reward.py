#the stage, set by hand: how much of the metabolic energy it pays (1 = all of it)
#0: off. A night at 0.5 with targets made it worse by its own score (+0.65 -> +0.04 per step, targets 1.2 -> 0.6,
#arms curling, still out of breath), not cheaper: with nothing working without the jet, keeping still was the least
#bad. Learn the targets and the floor first; later, if it wastes effort, a small effort cost like the benchmarks'
#(not the whole metabolic bill)
ENERGY_SHARE = 0.0


def reward(info):
    progress = info["speed"]
    #speed = how fast it moves in the episode's direction (m/s), negative going the other way
    #no speed cap: what can make going faster cost is the metabolic energy (off for now, ENERGY_SHARE = 0)
    #x10: moving that way is worth spending on (at x1 its progress was ~4% of its energy cost and it ignored it;
    #moving back and forth still nets 0)
    progress = 10 * progress

    jerk = ((info["action"] - info["previous_action"]) **2).mean()
    #no rules for height, bouncing or leaving the floor: in water swimming is allowed
    #spinning costs a little (spinning in place was an old cheat), capped so a flailing newborn
    #isn't fined into standing still
    turn = min(info["spin"] ** 2, 0.5)
    #no rule about facing the direction: a real octopus crawls any way
    #muscle effort costs points: flapping the tentacles fast or slamming them wastes energy
    #real animals move the way that spends the least energy, that's what makes them look natural
    #power is metabolic (what the food pays), per kg of octopus: the same effort costs the same for any body size
    #straight line: every W/kg costs the same, like food paid per joule. The old log went flat at high power
    #(at 580 W/kg, saving 100 W/kg was worth only 0.16), so flailing light arms barely cost more than calm ones
    #0.013 when paying all of it: a flailing newborn (~150 W/kg) ~1.9 per step, a good crawl (~2 W/kg) ~0.02
    #(the old cap at 1.0 made everything above ~1700 W free, and it learned to burn 3000 W)
    watts_per_kg = info["power"] / info["mass"]
    energy = ENERGY_SHARE * 0.013 * watts_per_kg
    #rigidity costs in full from birth: holding the same muscle command for a long time, not each contraction
    #(charging every contraction in full taught a newborn that moving its arms at all was too expensive)
    #0.9: rigid at 0.65 costs ~0.26 per step and a ball ~0.10, standing still 0: both cheats score below doing nothing
    rigidity = 0.9 * info["rigidity"]
    #upside down lying on the floor: swimming it can tilt any way (like a missile, as real octopuses do), but on the
    #floor upside down it can't crawl and its jet points at the floor. It jetted hard, flipped and lay there 70-80% of the time
    #airborne = nothing touches the floor (arms don't collide with each other), so tilted in the water is free
    #1.0: more than rolling along on its back earns (~0.46 per step): at 0.3 it still paid off, so it learned to curl up and
    #roll on its back to the targets. It still doesn't end the episode (dying early used to be a way out)
    upside_down = 1.0 * (info["flipped"] and not info["airborne"])
    #no prize for reaching the target: the trip there already pays ~400 per meter. A +300 prize put ~1/3 of the whole
    #learning signal on the 2% of steps just before an arrival, and the critic couldn't see it coming (back then the
    #brain got "which way", not "how far"): twice it drifted into curling up into a ball with it on
    #no rule about keeping the tips on the floor: in water swimming is a healthy octopus move
    score = progress - 0.05 * jerk - 0.05 * turn - energy - rigidity - upside_down

    return score
