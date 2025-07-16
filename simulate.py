import warnings
warnings.filterwarnings("ignore")

import os
import json
import random as rnd
import numpy as np
import torch
import torch.optim as optim
import wandb
import argparse

# Let's import our custom-built adventuring tools!
from DnDSimulator.Entity_class import *
from DnDSimulator.Encounter_Simulator import *
from DnDSimulator.Dm_class import *
from tools import (generate_enemy_vector, map_enemy_vector_to_names, 
                   generate_and_save_party_matrices, calculate_party_exp, 
                   calculate_enemy_exp)
from dm_policy import dm_guide_policy
# The REINFORCE classes are imported later, only if needed.
# This keeps our code clean and avoids unnecessary imports if we're just running a random policy.

def reward_function(win_probability, rounds_number, damage_player, DeathNumber, TeamHealth, party_size, repetitions, verbose=False):
    """
    Behold, the soul of our machine! This function determines what is "good" and "bad".
    It's a delicate alchemy of combat stats, designed to reward exciting, challenging,
    and ultimately winnable battles. A high reward means the AI created a masterpiece of an encounter.
    A low reward means it was either a snooze-fest or a massacre.

    :param win_probability: The party's win rate (0.0 to 1.0). Winning is good. Very good.
    :param rounds_number: Average number of rounds the combat lasted. Quick, decisive battles are preferred.
    :param damage_player: Total damage dealt by the party. We want our heroes to feel heroic!
    :param DeathNumber: How many heroes bit the dust. A little danger is exciting, but too much is a tragedy.
    :param TeamHealth: The average remaining health of the party. We want them to break a sweat!
    :param party_size: The number of heroes in the party.
    :param repetitions: How many times the simulation was run.
    :param verbose: If True, prints a detailed breakdown of the reward calculation.
    :return: A single float value representing the total reward. Higher is better!
    """
    # A massive reward for winning. This is the main goal, after all.
    win_reward = win_probability * 1000
    # We like snappy combat. A small bonus for each round, encouraging engagement.
    combat_duration_reward = rounds_number * 5
    # A party that finishes with full health was never in danger. We reward taking some damage.
    missing_health_reward = (1 - TeamHealth) * 100
    # Every hero's life is precious. We add a small reward for each fallen hero to encourage risk.
    deaths_reward = sum(DeathNumber)
    # The ultimate failure: a Total Party Kill (TPK). This incurs a soul-crushing penalty.
    tpk_penalty = -10000 if sum(DeathNumber) >= party_size * repetitions else 0
    # Let the damage numbers fly! A small reward for the total damage dealt.
    damage_reward = sum(damage_player) * 5
    
    total_reward = (win_reward + combat_duration_reward + missing_health_reward + 
                    deaths_reward + tpk_penalty + damage_reward)

    if verbose:
        print("\n--- 🧙 Reward Calculation ---")
        print(f"  Win Reward: {win_reward:.2f}")
        print(f"  Duration Reward: {combat_duration_reward:.2f}")
        print(f"  Missing Health Reward: {missing_health_reward:.2f}")
        print(f"  Deaths Reward: {deaths_reward:.2f}")
        print(f"  TPK Penalty: {tpk_penalty:.2f}")
        print(f"  Damage Reward: {damage_reward:.2f}")
        print(f"--------------------------")
        print(f"  Total Reward: {total_reward:.2f}")
        print("--------------------------")

    return total_reward

def benchmark(party, enemy_names, verbose=False):
    """
    Welcome to the Thunderdome! This function is the combat simulator.
    It takes a party of heroes and a list of enemies, then runs a statistically
    significant number of battles (100, to be precise) to see what happens.
    It's a digital crucible for testing encounter balance.

    :param party: A list of hero class names.
    :param enemy_names: A list of monster names.
    :param verbose: If True, the simulation will print every dice roll. Not for the faint of heart.
    :return: The final reward and a tuple containing all the juicy combat statistics.
    """
    DM = DungeonMaster() # Every game needs a Dungeon Master!
    if not verbose:
        DM.block_print() # We tell the DM to be quiet so we're not flooded with text.

    # Let's set up the board. Team 0 for the heroes, Team 1 for the monsters.
    Entities = {i: {"name": f"{member} Lv5", "team": 0} for i, member in enumerate(party)}
    enemy_start_index = len(party)
    for i, enemy in enumerate(enemy_names):
        if enemy is not None:
            Entities[enemy_start_index + i] = {"name": enemy, "team": 1}

    # Create the actual fighter objects from the entity definitions.
    Fighters = [entity(Entities[i]['name'], Entities[i]['team'], DM, archive=True) for i in Entities]
    
    # Let the games begin! Run 100 simulations and gather the results.
    text, win_prob, rounds, dmg, deaths, health = full_statistical_recap(100, Fighters)
    
    # Now, let's score the encounter using our reward function.
    reward = reward_function(
        win_prob, np.mean(rounds), dmg, deaths, np.mean(health),
        party_size=len(party), repetitions=100, verbose=verbose
    )
    
    return reward, (win_prob, rounds, dmg, deaths, health)

# ===============================================================
# Main Training Loop - This is where the magic happens!
# ===============================================================
if __name__ == '__main__':
    # Using argparse to create a user-friendly command-line interface.
    # It's like creating a character sheet for our script!
    parser = argparse.ArgumentParser(description="🧙 Train a reinforcement learning agent to be a Dungeon Master! 🐉")
    
    # --- The Fun Settings ---
    parser.add_argument("--policy", type=str, default="reinf", choices=["reinf", "rand", "dm"], help="Choose your champion: 'reinf' (the AI), 'rand' (chaos), or 'dm' (a wise old heuristic).")
    parser.add_argument("--steps", type=int, default=1000, help="How many encounters to simulate and learn from.")
    parser.add_argument("--seed", type=int, default=42, help="The answer to life, the universe, and reproducibility.")
    parser.add_argument("--party_path", type=str, default="DnDSimulator/Classes", help="Path to the hero character sheets (JSONs).")
    parser.add_argument("--enemy_path", type=str, default="DnDSimulator/Enemies", help="Path to the monster stat blocks (JSONs).")
    parser.add_argument("--difficulty", type=str, default="deadly", choices=["easy", "medium", "hard", "deadly"], help="How much of a challenge should this be?")
    
    # --- The Boring-but-Important Settings ---
    parser.add_argument("--hidden_dim", type=int, default=16, help="The size of the hidden layers in our neural network's brain.")
    parser.add_argument("--pre_path", type=str, default="", help="Path to pre-made party data. If empty, we'll generate it on the fly.")  
    parser.add_argument("--device", type=int, default=0, help="Which GPU to use, if you have one. 0 is usually the first one.")
    parser.add_argument("--no_wandb", action="store_true", help="Don't log this run to Weights & Biases.")
    parser.add_argument("--layer_norm", action="store_true", help="Use Layer Normalization to keep the AI's thoughts stable.")
    parser.add_argument("--save_path", type=str, default="", help="Where to save the trained AI models.")
    args = parser.parse_args()

    # Set up the hardware and random seeds. This ensures our experiments are repeatable.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Let's get this adventure started on device: {device}")
    np.random.seed(args.seed)
    rnd.seed(args.seed)
    torch.manual_seed(args.seed)

    # Naming our experiment so we can find it later.
    exp_name = f"{args.policy.upper()}_{args.difficulty}_{args.seed}"

    # Initialize Weights & Biases for logging, unless told otherwise.
    # It's like having a scribe to record every detail of our adventure.
    if not args.no_wandb:
        wandb.init(project="NTRL_DM", name=exp_name, config=args)

    # Load pre-generated party data or create it from scratch.
    if not args.pre_path:
        print("⚡ Generating party matrices on the fly... this might take a moment.")
        party_data = generate_and_save_party_matrices(
            n=args.steps, min_players=3, max_players=8,
            class_files_path=args.party_path, save_path="", base_seed=args.seed
        )
        precomputed_parties, precomputed_class_names, index_ranges = party_data
    else:
        # Code to load from pre_path...
        print(f"✅ Loading precomputed party matrices from {args.pre_path}...")
        data = np.load(args.pre_path, allow_pickle=True)
        precomputed_parties = data["matrices"]
        precomputed_class_names = data["class_names"]
        # We still need to extract index_ranges
        from tools import extract_feature_constants
        _, _, _, _, index_ranges, _ = extract_feature_constants(args.party_path)

    party_indices = list(range(len(precomputed_parties)))
    rnd.shuffle(party_indices)

    def get_next_party_matrix():
        """A trusty function to grab the next party from our shuffled list."""
        global party_indices
        if not party_indices: # If we've used all parties, reshuffle and start over.
            party_indices = list(range(len(precomputed_parties)))
            rnd.shuffle(party_indices)
        index = party_indices.pop()
        return precomputed_parties[index], precomputed_class_names[index]

    # Get a list of all possible monsters from our monster manual.
    enemy_list = sorted([f.replace(".json", "") for f in os.listdir(args.enemy_path) if f.endswith(".json")])

    # If we're using the REINFORCE policy, we need to build our AI.
    if args.policy == "reinf":
        from reinforce import Policy, sample_enemies, reinforce_update
        print("🧠 Building the AI Dungeon Master...")
        policy_net = Policy(
            index_ranges=index_ranges,
            num_classes=len(enemy_list),
            hidden_dim=args.hidden_dim,
            layer_norm=args.layer_norm
        ).to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
        if not args.no_wandb:
            wandb.watch(policy_net, log="all", log_freq=10)
    
    rewards = []
    best_reward = -np.inf

    # The main training loop! Each iteration is one full cycle of picking a party,
    # creating an encounter, simulating it, and learning from the result.
    print(f"⚔️  Training the AI Dungeon Master for {args.steps} steps:")
    for i in range(args.steps):
        # 1. Get a party of heroes
        party_matrix, party_classes = get_next_party_matrix()
        party_tensor = torch.tensor(party_matrix, dtype=torch.float32).to(device).unsqueeze(0)
        party_exp = calculate_party_exp(party_classes, difficulty=args.difficulty)
        
        # 2. Generate an enemy encounter based on the chosen policy
        log_probs = [] # We only need this for the 'reinf' policy
        if args.policy == "reinf":
            chosen_indices, log_probs = sample_enemies(policy_net, party_tensor, device=device)
            enemy_names = map_enemy_vector_to_names(chosen_indices, enemy_list)
        elif args.policy == "rand":
            enemy_array = generate_enemy_vector(3, 8, args.enemy_path)
            enemy_names = map_enemy_vector_to_names(enemy_array, enemy_list)
            enemy_names = [e for e in enemy_names if e != -1] # Clean up placeholders
        elif args.policy == "dm":
            enemy_names = dm_guide_policy(party_exp)
        
        # 3. Run the simulation and get the reward
        reward, stats = benchmark(party_classes, enemy_names)
        rewards.append(reward)

        # 4. If we're training the AI, update its brain
        if args.policy == "reinf":
            reinforce_update(log_probs, reward, optimizer, device=device)

        # 5. Log our progress
        print(f"Step {i+1}/{args.steps} 🎲 | Reward: {reward:.2f}", end='\r', flush=True)

        if not args.no_wandb and (i + 1) % 10 == 0:
            win_prob, rounds, _, _, _ = stats
            log_data = {
                "step": i + 1,
                "reward": reward,
                "avg_reward_last_100": np.mean(rewards[-100:]),
                "win_probability": win_prob,
                "avg_rounds": np.mean(rounds),
                "encounter_size": len(enemy_names)
            }
            wandb.log(log_data)
        
        # 6. Save the best model we've found so far
        if args.policy == "reinf" and reward > best_reward:
            best_reward = reward
            if args.save_path:
                local_save_dir = os.path.join(args.save_path, exp_name)
                os.makedirs(local_save_dir, exist_ok=True)
                best_model_path = os.path.join(local_save_dir, "pi_best.pth")
                torch.save(policy_net.state_dict(), best_model_path)
                print(f"\n🐉 New Best Model! Reward: {best_reward:.2f}. Saved to {best_model_path}")

    print(f"\n\n🧙 The grand adventure concludes! Final Average Reward: {np.mean(rewards):.2f} 🎲")