# NTRL 🧙‍♂️🐉🎲

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2506.19530)



Tired of your DM's "deadly" encounters being about as threatening as a fluffy kitten? Or the opposite, where a "medium" challenge accidentally turns into a TPK because the rulebook lied? This project is for you.

We've created an AI that went to the school of hard knocks—by running thousands of simulated D&D battles—to learn how to be the perfect Dungeon Master for combat. It designs challenging, tactical, and most importantly, *fun* encounters automatically.

---
## How It Works 

NTRL doesn't just count XP. It learned its craft using a **Reinforcement Learning** technique called **REINFORCE**. We framed the task as a **Contextual Bandit** problem, which is a fancy way of saying it's like a club bouncer who sizes up your party before deciding how many goblins to let inside.

The AI was trained by maximizing a special **reward function**—its personal philosophy on what makes combat awesome:
* ⚔️ **Longevity is Key**: A great battle should be a multi-round epic, not a one-hit wonder.
* 🩸 **Bring the Pain**: The party should feel the heat! It's rewarded for dealing damage and making players sweat (and use those healing potions).
* ☠️ **Calculated Risks**: The AI aims for thrilling encounters where death is a possibility, but it gets a massive penalty for a **Total Party Kill (TPK)**. It wants drama, not a tragedy.

---
## Results

So, what happens when you let a robot run the show? Glorious, well-balanced chaos.
* 📖 **Epic Sagas**: Fights last **+200%** longer, giving everyone a chance to shine.
* 🎯 **Real Scrapes**: Parties finish with **-16.67%** fewer hit points, making victory feel earned.
* ⚖️ **Always Fair**: It maintains a consistently high **~70% win rate**. It's tough, not a cheater.

---
## Getting Started

Ready to summon your own AI DM? Just follow these simple steps.

Clone this repository to your local machine.
```bash
git clone [https://github.com/your-username/NTRL.git](https://github.com/your-username/NTRL.git)
cd NTRL
pip install -r requirements.txt
python simulate.py [OPTIONS]
```

### Common Options:

--policy: The strategy to use. Choose from reinf (the NTRL agent), rand (random selection), or dm (DMG heuristics).

--steps: The number of encounters to generate and simulate.

--difficulty: The target encounter difficulty (easy, medium, hard, deadly).

--no_wandb: Disables logging to Weights & Biases.

### Example:

```bash
# Run 5000 simulation steps using the NTRL agent with a 'deadly' difficulty target
python simulate.py --policy reinf --difficulty deadly --steps 5000
```
---
## Citation
If you use the code or ideas from this project in your research, please cite our paper:


```bash
@article{romeo2025ntrl,
  title={{NTRL: Encounter Generation via Reinforcement Learning for Dynamic Difficulty Adjustment in Dungeons and Dragons}},
  author={Romeo, Carlo and Bagdanov, Andrew D.},
  booktitle={{IEEE Conference on Games (CoG)}},
  year={2025}
}
```

---
## Acknowledgments
This code leverages the **AMAZING** Dungeons & Dragons Combat Simulator created by DanielK314. The simulated combat environment was essential for training our agent. You can find the original repository here: [https://github.com/DanielK314/DnDSimulator.git](https://github.com/DanielK314/DnDSimulator.git)
