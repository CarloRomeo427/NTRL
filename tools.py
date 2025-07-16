import json
import os
import random as rnd
import numpy as np

def calculate_party_exp(party, difficulty="hard"):
    """
    This function is our "Difficulty-o-Meter." Given a party of brave adventurers,
    it calculates their total experience point (XP) threshold for a given difficulty.
    Think of it as setting the bar for how much of a challenge the party can handle
    before things get... well, deadly.

    :param party: A list of party members. We just need the count, really.
    :param difficulty: How much trouble are we looking for? Can be "easy", "medium", "hard", or "deadly".
    :return: The total XP budget for the encounter.
    """
    # These are the sacred numbers from the scrolls of the Dungeon Master's Guide.
    # They tell us how much XP per character corresponds to each difficulty level for a Level 5 party.
    EXP_THRESHOLDS = {
        "easy": 250,
        "medium": 500,
        "hard": 750,
        "deadly": 1100,
    }
    party_size = len(party)

    # Simple math: multiply the per-character XP threshold by the number of characters. Voila!
    return EXP_THRESHOLDS[difficulty] * party_size

# The official "Action Economy" multiplier table. This tells you that fighting
# two goblins is more than twice as hard as fighting one. It's a cruel, cruel world.
MULTIPLIER_TABLE = {
    1: 1.0, 2: 1.5, 3: 2.0, 4: 2.0, 5: 2.0, 6: 2.0, 7: 2.5, 8: 2.5,
    9: 2.5, 10: 2.5, 11: 3.0, 12: 3.0, 13: 3.0, 14: 3.0, 15: 4.0,
}

def calculate_enemy_exp(enemy_list):
    """
    Calculates the *actual* challenge rating of an enemy group. It's not just the sum
    of their XP; it's adjusted for how many of them there are. More enemies means more
    chaos, and this function quantifies that chaos in XP form.

    :param enemy_list: A list of enemy names to be appraised.
    :return: The total, multiplier-adjusted XP value of the encounter.
    """
    # Our big book of monsters and their base XP values.
    exp_dict = {
        "Ape": 100, "Boar": 50, "Brown Bear": 200, "Crocodile": 100, "Displacer Beast": 700,
        "Fire Elemental": 1800, "Flameskull": 1100, "Giant Boar": 450, "Giant Centipede": 50,
        "Giant Crocodile": 1800, "Giant Eagle": 200, "Giant Scorpion": 700, "Giant Spider": 200,
        "Giant Wasp": 100, "Goblin": 50, "Night Hag": 1800, "Ogre": 450, "Pirate": 200,
        "Polar Bear": 450, "Stone Giant": 2900, "Swarm of Bats": 50, "Vampire Spawn": 1800,
        "Vampire": 10000, "Wolf": 50, "Young Dragon": 5900,
    }

    # Tally up the base XP for all the baddies in the list.
    xp_values = [exp_dict.get(enemy, 0) for enemy in enemy_list]
    base_xp = sum(xp_values)

    # Now, let's see how many there are and apply the appropriate multiplier.
    enemy_count = len(xp_values)
    multiplier = MULTIPLIER_TABLE.get(enemy_count, 4.0) # Default to x4 if they brought the whole village.

    return base_xp * multiplier

def generate_enemy_vector(min_enemies, max_enemies, enemy_files_path):
    """
    Creates a random lineup of monsters for the party to face. It's like a gacha machine,
    but instead of cute characters, you get things that want to eat you. It also pads the
    list to a fixed size of 8 with -1s, because neural networks love fixed-size inputs.

    :param min_enemies: The smallest number of enemies we're willing to throw at the party.
    :param max_enemies: The absolute maximum number of enemies.
    :param enemy_files_path: The folder where the monster stat blocks (.json files) live.
    :return: A numpy array of length 8, filled with enemy indices or -1 for empty slots.
    """
    enemy_files = [f for f in os.listdir(enemy_files_path) if f.endswith(".json")]
    num_enemies = len(enemy_files)

    if num_enemies == 0:
        raise ValueError("The monster manual is empty! We can't have a fight without monsters.")

    # Let's roll a die to see how many monsters show up today.
    num_selected_enemies = rnd.randint(min_enemies, max_enemies)

    # Now, let's randomly pick which monsters are coming to the party.
    # We use `choices` to allow for duplicates (e.g., a pack of wolves).
    selected_indices = rnd.choices(range(num_enemies), k=num_selected_enemies)

    # Our model expects a vector of exactly 8. If we have fewer, we fill the rest with -1.
    # It's like saying "no monster here" in a way the computer understands.
    while len(selected_indices) < 8:
        selected_indices.append(-1)

    return np.array(selected_indices)

def map_enemy_vector_to_names(enemy_vector, enemy_list):
    """
    Translates a vector of cryptic enemy indices back into something a human can read,
    like "Goblin" or "Young Dragon". It's our universal translator for monster IDs.
    It also handles a special "STOP" signal, which tells our model when to stop picking enemies.

    :param enemy_vector: The numpy array of enemy indices from `generate_enemy_vector`.
    :param enemy_list: The master list of all possible enemy names.
    :return: A list of human-readable enemy names.
    """
    enemy_names = []
    for idx in enemy_vector:
        if idx == len(enemy_list):
            # This is our special "That's enough monsters for today" signal.
            enemy_names.append("STOP")
        elif idx != -1: # We ignore the -1 placeholders.
            enemy_names.append(enemy_list[idx])
    return enemy_names

def extract_feature_constants(class_files_path):
    """
    This function is like a librarian for our character classes. It reads all the hero JSON files
    and figures out all the unique features, their maximum lengths, and how to organize them.
    This metadata is crucial for turning a character sheet into a numerical vector our model can eat.

    :param class_files_path: The directory where all the character class JSONs are stored.
    :return: A treasure trove of metadata: lists of features, their max lengths, unique values,
             and the index ranges for slicing them up.
    """
    # These are the stats we care about. The building blocks of any great hero or villain.
    numerical_features = [
        "AC", "HP", "Proficiency", "To_Hit", "Attacks", "DMG",
        "Str", "Dex", "Con", "Int", "Wis", "Cha",
        "Spell_DC", "Spell_Mod", "Spell_Slot_1", "Spell_Slot_2", "Spell_Slot_3",
        "Speed", "Range_Attack"
    ]
    # We group categorical features for easier handling. It's all about organization!
    cat_set_1 = ["Saves_Proficiency", "Damage_Type", "Position"]
    cat_set_2 = ["Spell_List"]
    cat_set_3 = ["Other_Abilities"]
    categorical_features = cat_set_1 + cat_set_2 + cat_set_3

    # We'll need to know the longest possible list for each categorical feature to create fixed-size vectors.
    categorical_max_lengths = {feature: 0 for feature in categorical_features}
    # And we need a set of all possible unique values for one-hot encoding.
    categorical_value_sets = {feature: set() for feature in categorical_features}

    # Let's read every character file to gather this information.
    for file_name in os.listdir(class_files_path):
        if file_name.endswith(".json"):
            with open(os.path.join(class_files_path, file_name), "r") as f:
                data = json.load(f)
                for feature in categorical_features:
                    values = str(data.get(feature, "")).split()
                    categorical_max_lengths[feature] = max(categorical_max_lengths[feature], len(values))
                    categorical_value_sets[feature].update(values)

    # Now we define the start and end indices for each feature set in our final vector.
    # This is like creating a map of our feature array.
    num_feature_count = len(numerical_features)
    cat_set_1_count = sum(categorical_max_lengths[f] for f in cat_set_1)
    cat_set_2_count = sum(categorical_max_lengths[f] for f in cat_set_2)
    cat_set_3_count = sum(categorical_max_lengths[f] for f in cat_set_3)

    index_ranges = {
        "numerical": (0, num_feature_count),
        "cat_set_1": (num_feature_count, num_feature_count + cat_set_1_count),
        "cat_set_2": (num_feature_count + cat_set_1_count, num_feature_count + cat_set_1_count + cat_set_2_count),
        "cat_set_3": (num_feature_count + cat_set_1_count + cat_set_2_count,
                       num_feature_count + cat_set_1_count + cat_set_2_count + cat_set_3_count)
    }

    return (numerical_features, categorical_features, categorical_max_lengths,
            {key: list(values) for key, values in categorical_value_sets.items()},
            index_ranges, (cat_set_1, cat_set_2, cat_set_3))


def precompute_class_arrays(class_files_path, numerical_features, categorical_features,
                            categorical_max_lengths, categorical_value_sets, index_ranges, cat_sets):
    """
    This function does the heavy lifting of converting all character JSON files into
    numerical numpy arrays. It's a "vectorizer" for our heroes. We do this once
    to save time during training.

    :param ...: All the metadata we gathered from `extract_feature_constants`.
    :return: A dictionary mapping class names to their shiny new numerical arrays.
    """
    cat_set_1, cat_set_2, cat_set_3 = cat_sets
    # The total length of our feature vector is determined by the end of the last index range.
    max_array_length = index_ranges["cat_set_3"][1]
    class_arrays = {}

    for file_name in os.listdir(class_files_path):
        if file_name.endswith(".json"):
            class_name = file_name.replace(" Lv5.json", "")
            file_path = os.path.join(class_files_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Create an empty canvas (a numpy array of zeros) to paint our features onto.
            feature_array = np.zeros(max_array_length, dtype=float)

            # First, the easy part: fill in the numerical features.
            num_start, _ = index_ranges["numerical"]
            for i, feature in enumerate(numerical_features):
                feature_array[num_start + i] = data.get(feature, 0)

            # Now for the trickier categorical features. We'll use our precomputed metadata.
            for feature_set, set_key in zip([cat_set_1, cat_set_2, cat_set_3],
                                            ["cat_set_1", "cat_set_2", "cat_set_3"]):
                start_idx, _ = index_ranges[set_key]
                offset = 0
                for feature in feature_set:
                    values = str(data.get(feature, "")).split()
                    possible_values = categorical_value_sets[feature]
                    for value in values[:categorical_max_lengths[feature]]:
                        if value in possible_values:
                            # We store the index of the value in our unique value list.
                            feature_array[start_idx + offset] = possible_values.index(value)
                        else:
                            # If we encounter an unknown value, we mark it with -1.
                            feature_array[start_idx + offset] = -1
                        offset += 1
            class_arrays[class_name] = feature_array
    return class_arrays


def create_party_matrix(selected_classes, class_arrays, max_array_length):
    """
    Assembles a party matrix from a list of selected class names. This matrix is what
    our neural network will see. It's a stack of up to 8 character vectors, padded with
    -1s if the party is smaller than 8.

    :param selected_classes: A list of class names in the party.
    :param class_arrays: The dictionary of precomputed class vectors.
    :param max_array_length: The length of a single character vector.
    :return: A (8, max_array_length) numpy matrix representing the party.
    """
    # We initialize a matrix of -1s. This represents empty slots in the party.
    party_matrix = np.full((8, max_array_length), -1, dtype=float)
    for i, class_name in enumerate(selected_classes[:8]):
        if class_name in class_arrays:
            # We look up the precomputed vector for each class and place it in the matrix.
            party_matrix[i] = class_arrays[class_name]
    return party_matrix


def generate_random_party(min_players, max_players, class_list):
    """
    Generates a random party of adventurers. It's like rolling up a new group of friends
    for a one-shot adventure, but much, much faster.

    :param min_players: The minimum number of players in the party.
    :param max_players: The maximum number of players.
    :param class_list: A list of all available character classes.
    :return: A list of randomly selected class names.
    """
    if not class_list:
        raise ValueError("We can't form a party without any heroes! The class list is empty.")
    # How many heroes are we taking on this adventure? Let's roll for it.
    num_players = rnd.randint(min_players, max_players)
    # Now, let's pick our heroes. We use `choices` to allow for multiple of the same class.
    return rnd.choices(class_list, k=num_players)


def generate_and_save_party_matrices(n, min_players, max_players, class_files_path, save_path, base_seed=42):
    """
    A utility function that automates the whole process: from extracting features, to
    precomputing arrays, to generating `n` random party matrices. It can also save
    the results to a file so you don't have to do it all over again.

    :param n: The number of random party matrices to generate.
    :param ...: Other parameters for party generation.
    :param save_path: Where to save the generated matrices. If empty, it just returns them.
    :return: The generated party matrices, their class names, and the index ranges.
    """
    # Step 1: Gather all the metadata about our classes.
    (numerical_features, categorical_features, categorical_max_lengths,
     categorical_value_sets, index_ranges, cat_sets) = extract_feature_constants(class_files_path)
    # Step 2: Convert all our classes into numerical arrays.
    class_arrays = precompute_class_arrays(class_files_path, numerical_features,
                                           categorical_features, categorical_max_lengths,
                                           categorical_value_sets, index_ranges, cat_sets)
    max_array_length = index_ranges["cat_set_3"][1]
    class_list = list(class_arrays.keys())
    party_matrices = []
    party_class_names = []
    # Step 3: Generate `n` random parties and their matrices.
    for i in range(n):
        rnd.seed(base_seed + i) # Use a different seed for each party for variety.
        selected_classes = generate_random_party(min_players, max_players, class_list)
        matrix = create_party_matrix(selected_classes, class_arrays, max_array_length)
        party_matrices.append(matrix)
        party_class_names.append(selected_classes)
    # Step 4 (Optional): Save our hard work to a file.
    if save_path:
        np.savez(save_path,
                 matrices=np.array(party_matrices),
                 class_names=np.array(party_class_names, dtype=object))
        print(f"✅ Hooray! {n} party matrices have been safely stored at {save_path}")
    return party_matrices, party_class_names, index_ranges