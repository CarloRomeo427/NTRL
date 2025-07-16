import json
import math

class CharacterGenerator:
    # Class-level dictionary for base stats
    base_classes = {
        "Barbarian": {
            "AC": 14, "HP": 14, "Proficiency": 2, "To_Hit": 5.0, "Attacks": 1, "DMG": 6, "Level": 1,
            "Str": 15, "Dex": 12, "Con": 14, "Int": 8, "Wis": 10, "Cha": 10,
            "Saves_Proficiency": "Str Con", "Speed": 30, "Position": "front",
            "Damage_Type": "slashing", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_DC": 0, "Spell_Mod": 0, "Spell_List": "none",
            "Other_Abilities": "Rage RecklessAttack", 
            "RageDmg": 2.0
        },
        "Bard": {
            "AC": 14, "HP": 10, "Proficiency": 2, "To_Hit": 4.0, "Attacks": 1, "DMG": 4, "Level": 1,
            "Str": 8, "Dex": 14, "Con": 12, "Int": 10, "Wis": 12, "Cha": 16,
            "Saves_Proficiency": "Dex Cha", "Speed": 30, "Position": "back",
            "Damage_Type": "slashing", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_DC": 13, "Spell_Mod": 3, "Spell_List": "ViciousMockery CureWounds",
            "Other_Abilities": "BardicInspiration"
        },
        "Cleric": {
            "AC": 16, "HP": 10, "Proficiency": 2, "To_Hit": 4.0, "Attacks": 1, "DMG": 6, "Level": 1,
            "Str": 14, "Dex": 10, "Con": 12, "Int": 10, "Wis": 16, "Cha": 12,
            "Saves_Proficiency": "Wis Cha", "Speed": 30, "Position": "front",
            "Damage_Type": "radiant", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_DC": 13, "Spell_Mod": 3, "Spell_List": "SacredFlame CureWounds",
            "Other_Abilities": "TurnUndead ChannelDivinity"
        },
        "Druid": {
            "AC": 14, "HP": 10, "Proficiency": 2, "To_Hit": 4.0, "Attacks": 1, "DMG": 4, "Level": 1,
            "Str": 10, "Dex": 12, "Con": 14, "Int": 10, "Wis": 16, "Cha": 12,
            "Saves_Proficiency": "Int Wis", "Speed": 30, "Position": "middle",
            "Damage_Type": "bludgeoning", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_DC": 13, "Spell_Mod": 3, "Spell_List": "Shillelagh Entangle",
            "Other_Abilities": "WildShape"
        },
        "Fighter": {
            "AC": 16, "HP": 12, "Proficiency": 2, "To_Hit": 6.0, "Attacks": 1, "DMG": 8, "Level": 1,
            "Str": 16, "Dex": 12, "Con": 14, "Int": 10, "Wis": 10, "Cha": 10,
            "Saves_Proficiency": "Str Con", "Speed": 30, "Position": "front",
            "Damage_Type": "slashing", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_List": "none", "Other_Abilities": "SecondWind"
        },
        "Monk": {
            "AC": 15, "HP": 10, "Proficiency": 2, "To_Hit": 5.0, "Attacks": 1, "DMG": 4, "Level": 1,
            "Str": 10, "Dex": 16, "Con": 14, "Int": 10, "Wis": 14, "Cha": 12,
            "Saves_Proficiency": "Str Dex", "Speed": 30, "Position": "middle",
            "Damage_Type": "bludgeoning", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_List": "none", "Other_Abilities": "FlurryOfBlows"
        },
        "Paladin": {
            "AC": 18, "HP": 12, "Proficiency": 2, "To_Hit": 6.0, "Attacks": 1, "DMG": 8, "Level": 1,
            "Str": 16, "Dex": 10, "Con": 14, "Int": 8, "Wis": 12, "Cha": 14,
            "Saves_Proficiency": "Wis Cha", "Speed": 30, "Position": "front",
            "Damage_Type": "radiant", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_DC": 12, "Spell_Mod": 2, "Spell_List": "LayOnHands",
            "Other_Abilities": "DivineSmite"
        },
        "Ranger": {
            "AC": 15, "HP": 12, "Proficiency": 2, "To_Hit": 6.0, "Attacks": 1, "DMG": 6, "Level": 1,
            "Str": 12, "Dex": 16, "Con": 14, "Int": 10, "Wis": 12, "Cha": 10,
            "Saves_Proficiency": "Str Dex", "Speed": 30, "Position": "middle",
            "Damage_Type": "piercing", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_DC": 12, "Spell_Mod": 2, "Spell_List": "HuntersMark",
            "Other_Abilities": "FavoredEnemy"
        },
        "Rogue": {
            "AC": 15, "HP": 10, "Proficiency": 2, "To_Hit": 5.0, "Attacks": 1, "DMG": 6, "Level": 1,
            "Str": 10, "Dex": 16, "Con": 14, "Int": 12, "Wis": 10, "Cha": 12,
            "Saves_Proficiency": "Dex Int", "Speed": 30, "Position": "middle",
            "Damage_Type": "slashing", "Damage_Resistance": "none", "Damage_Immunity": "none",
            "Spell_List": "none", "Other_Abilities": "SneakAttack"
        }
    }

    def __init__(self, class_name, level):
        self.class_name = class_name
        self.level = level

        # Validate the class name
        if class_name not in self.base_classes:
            raise ValueError(f"Error: Class '{class_name}' not found.")

        # Copy base stats so we don't overwrite the dictionary at the class level
        self.base_stats = self.base_classes[class_name].copy()

        # Generate the scaled stats based on level
        self.scaled_stats = self.level_up_stats(self.base_stats, self.level)

    def level_up_stats(self, base_stats, level):
        """Progress character stats and abilities according to level scaling."""
        new_stats = base_stats.copy()
        new_stats["Level"] = level  # Store the current level

        # Increase HP using hit dice average:
        #   if the class's base HP is (die_max + CON) at level 1,
        #   we subtract the CON bonus to get the die_max,
        #   then each subsequent level adds average HP (die_max // 2 + 1).
        hit_die = (base_stats["HP"] - 2)  # e.g., Barbarian has 14 HP = d12(12) + CON(2)
        avg_hp_gain = (hit_die // 2) + 1
        # Add this average HP for each additional level beyond 1
        new_stats["HP"] += avg_hp_gain * (level - 1)

        # Increase Proficiency Bonus according to standard 5e progression
        # Level 1-4 => +2, 5-8 => +3, 9-12 => +4, 13-16 => +5, 17-20 => +6
        new_stats["Proficiency"] = 2 + (level - 1) // 4

        # Extra Attacks at levels 5, 11, and 20 (typical for martial classes)
        if level >= 5:
            new_stats["Attacks"] = 2.0
        if level >= 11:
            new_stats["Attacks"] = 3.0
        if level == 20:
            new_stats["Attacks"] = 4.0

        # Example: Barbarian's Rage damage scaling (if present)
        if "RageDmg" in new_stats:
            # Simple approach: increase rage damage by +1 at levels 9 and 16, etc.
            # Here, as an example: 2 at L1 -> 3 at L9 -> 4 at L17
            new_stats["RageDmg"] = 2 + (level // 8)

        # Example: Rogue's Sneak Attack damage scaling (if present)
        if "Other_Abilities" in new_stats and "SneakAttack" in new_stats["Other_Abilities"]:
            # Typically 1d6 at L1, but you can scale as you wish
            # This is a placeholder example: +1d6 every two levels
            new_stats["Sneak_Attack_Dice"] = (level + 1) // 2  # e.g., 10th level => 5d6

        return new_stats

    def get_stats(self):
        """Return the dictionary of scaled stats so it can be JSON-serialized."""
        return self.scaled_stats


# ---------------------------
# Example usage / test script
# ---------------------------
if __name__ == "__main__":
    # Create a 20th-level Fighter
    fighter_20 = CharacterGenerator("Fighter", 20)

    # Convert the stats to JSON-serializable dictionary and print nicely
    print(json.dumps(fighter_20.get_stats(), indent=4))
