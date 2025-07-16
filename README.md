# NTRL: The AI Dungeon Master 🧙‍♂️🐉🎲

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Tired of your DM's "deadly" encounters being about as threatening as a fluffy kitten? Or the opposite, where a "medium" challenge accidentally turns into a TPK because the rulebook lied? This project is for you.

We've created an AI that went to the school of hard knocks—by running thousands of simulated D&D battles—to learn how to be the perfect Dungeon Master for combat. It designs challenging, tactical, and most importantly, *fun* encounters automatically.

---
## How It Works (It's not Skynet, we promise)

NTRL doesn't just count XP. It learned its craft using a **Reinforcement Learning** technique called **REINFORCE**. We framed the task as a **Contextual Bandit** problem, which is a fancy way of saying it's like a club bouncer who sizes up your party before deciding how many goblins to let inside.

The AI was trained by maximizing a special **reward function**—its personal philosophy on what makes combat awesome:
* ⚔️ **Longevity is Key**: A great battle should be a multi-round epic, not a one-hit wonder.
* 🩸 **Bring the Pain**: The party should feel the heat! It's rewarded for dealing damage and making players sweat (and use those healing potions).
* ☠️ **Calculated Risks**: The AI aims for thrilling encounters where death is a possibility, but it gets a massive penalty for a **Total Party Kill (TPK)**. It wants drama, not a tragedy.

---
## The AI's Report Card 📈

So, what happens when you let a robot run the show? Glorious, well-balanced chaos.
* **Epic Sagas**: Fights last **+200%** longer, giving everyone a chance to shine.
* **Real Scrapes**: Parties finish with **-16.67%** fewer hit points, making victory feel earned.
* **Always Fair**: It maintains a consistently high **~70% win rate**. It's tough, not a cheater.

![Comparison charts from the paper showing NTRL (orange) kicking butt](https://i.imgur.com/k9vLz8Y.png)

---
## The Ritual: Getting Started

Ready to summon your own AI DM? Just follow these simple steps.

### 1. Gather Your Components
* Python 3.9+
* PyTorch, NumPy
* `pip install -r requirements.txt`

### 2. Scribe the Scroll
Clone this repository to your local machine.
```bash
# Clone this ancient tome
git clone [https://github.com/your-username/NTRL-AI-DM.git](https://github.com/your-username/NTRL-AI-DM.git)
cd NTRL-AI-DM
