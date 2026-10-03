#the stage, set by hand: how much of the metabolic energy it pays (1 = all of it)
#0 for the start of stage 1: a flailing newborn then pays only ~0.05 per step (jerk, spin, rigidity) against 0 for
#standing still, so it has no reason to freeze before it learns to crawl (at 0.2 it paid ~0.4, at 1 ~1.9).
#Raise it by hand once it crawls, so it learns to crawl cheaply
ENERGY_SHARE = 0.0


def reward(info):
    progress = info["speed"]
    #speed = how fast it moves in the episode's direction (m/s), negative going the other way
    #no speed cap: going faster is paid for with metabolic energy
    #x10: moving that way is worth spending on (at x1 its progress was ~4% of its energy cost and it ignored it;
    #moving back and forth still nets 0)
    progress = 10 * progress

    jerk = ((info["action"] - info["previous_action"]) **2).mean()
    #no rules for height, bouncing or leaving the floor: in water swimming is allowed,
    #and the metabolic energy decides if it's worth it
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
    #no rule about keeping the tips on the floor: in water swimming is a healthy octopus move
    score = progress - 0.05 * jerk - 0.05 * turn - energy - rigidity

    #no extra cost for flipping over, and it doesn't end the episode: upside down it can't crawl,
    #so the normal costs keep running until it rights itself

    return score
