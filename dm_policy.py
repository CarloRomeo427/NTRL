from itertools import combinations
import random

def dm_guide_policy(party_exp_threshold):
    """
    Ah, the Dungeon Master's secret weapon! This function plays the role of a meticulous DM,
    trying to build the "perfect" encounter. It sifts through all possible combinations of monsters
    to find a group whose total adjusted XP is as close as a hair's breadth to the party's
    XP threshold. It's like trying to perfectly toast a marshmallow without setting it on fire.

    :param party_exp_threshold: (int) The total XP budget our dear DM has to spend.
                                  Think of it as their monster-shopping allowance.
    :return: (list) A list of enemy names that represents the DM's best attempt at a balanced,
                    yet terrifying, encounter.
    """

    # Here's our menagerie of horrors and their corresponding XP price tags.
    # A DM's shopping catalog, if you will.
    enemy_xp_dict = {
        "Ape": 100,
        "Boar": 50,
        "Brown Bear": 200,
        "Crocodile": 100,
        "Displacer Beast": 700, # Note: Corrected from "Displayer"
        "Fire Elemental": 1800,
        "Flameskull": 1100,
        "Giant Boar": 450,
        "Giant Centipede": 50,
        "Giant Crocodile": 1800,
        "Giant Eagle": 200,
        "Giant Scorpion": 700,
        "Giant Spider": 200,
        "Giant Wasp": 100,
        "Goblin": 50,
        "Night Hag": 1800,
        "Ogre": 450,
        "Pirate": 200,
        "Polar Bear": 450,
        "Stone Giant": 2900,
        "Swarm of Bats": 50,
        "Vampire Spawn": 1800,
        "Vampire": 10000,
        "Wolf": 50,
        "Young Dragon": 5900,
    }

    # Let's turn our dictionary into a list of tuples (name, xp) for easier handling.
    # It's like putting all our monster trading cards in a neat binder.
    enemy_list = list(enemy_xp_dict.items())
    best_combination = []
    # We start by assuming the worst possible match, with an infinitely large XP difference.
    min_xp_diff = float("inf")

    # Time for some brute-force magic! We'll check every possible team size,
    # from a lonely single monster up to a terrifying mob of 8.
    for team_size in range(1, min(8, len(enemy_list)) + 1):
        # Using itertools.combinations is like having a superpower to see all possible teams
        # without breaking a sweat.
        for combo in combinations(enemy_list, team_size):
            # First, calculate the raw, unadjusted XP of the monster group.
            base_xp = sum(enemy[1] for enemy in combo)

            # Now for the DM's secret sauce: the XP multiplier! The more monsters, the deadlier the fight.
            # We fetch the multiplier from our sacred MULTIPLIER_TABLE.
            multiplier = MULTIPLIER_TABLE.get(team_size, 4.0)
            adjusted_xp = base_xp * multiplier
            # How close did we get to our target? Let's find out.
            xp_diff = abs(party_exp_threshold - adjusted_xp)

            # If this new combination is a better fit than our previous best, we have a new winner!
            # It's like finding a key that fits the lock just a little bit better.
            if xp_diff < min_xp_diff:
                min_xp_diff = xp_diff
                best_combination = combo

    # Once we've checked all possibilities, we return the names of the monsters in our "golden" encounter.
    # "Go forth, my minions, and make this a memorable fight!"
    return [enemy[0] for enemy in best_combination]

# This table is straight from the Dungeon Master's Guide. It tells us how much deadlier
# a group of monsters is compared to a single one. It’s the secret ingredient to TPKs.
MULTIPLIER_TABLE = {
    1: 1.0, 2: 1.5, 3: 2.0, 4: 2.0, 5: 2.0, 6: 2.0, 7: 2.5,
    8: 2.5, 9: 2.5, 10: 2.5, 11: 3.0, 12: 3.0, 13: 3.0, 14: 3.0, 15: 4.0
}