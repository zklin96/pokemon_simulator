# How the Pokemon VGC AI Works
## A Guide for Pokemon Players (No ML Background Required)

---

## 🎯 The Big Picture

Imagine you're teaching a brand new player to become a competitive VGC battler. You'd probably:

1. **Show them replays** of top players and say "do what they do"
2. **Have them practice** against bots to build muscle memory
3. **Let them play ladder** to face real opponents and improve

That's exactly what this AI does! It goes through three "training stages" that mirror how humans learn.

---

## 📚 Stage 1: Learning from the Pros (Imitation Learning)

### What happens:
The AI watches **21,000+ real VGC battles** from top players on Pokemon Showdown.

For each turn, the AI sees:
- What Pokemon are on the field
- HP, status conditions, stat boosts
- Weather, terrain, trick room, etc.
- What moves are available

Then it learns: **"In this situation, the expert chose THIS action."**

### Pokemon analogy:
It's like watching Wolfe Glick or Aaron Zheng VODs and memorizing:
> "When my Flutter Mane is facing Incineroar, and I have Moonblast available, the pros usually click Moonblast."

### What the AI learns:
- Basic threat recognition (what's dangerous?)
- Common plays (Fake Out turn 1, Protect on predicted double-up)
- Positioning (when to switch, what matchups to seek)

### Limitation:
The AI can copy plays but doesn't truly *understand* why. It's like a player who memorizes "always Protect Amoonguss when Incineroar is in" without understanding it's because of Flare Blitz.

---

## 🎮 Stage 2: Practice Mode (PPO Simulation)

### What happens:
The AI plays **hundreds of thousands of simulated battles** against itself at super-speed (~800 battles per second).

Instead of just copying experts, now the AI experiments and learns from results.

### The Reward System (How the AI knows what's good)

The AI gets "points" for good plays and loses points for bad ones:

| Action | Points | Why |
|--------|--------|-----|
| **Win the battle** | +10 | Ultimate goal |
| **Lose the battle** | -10 | Avoid this |
| **KO an opponent** | +2 | Progress toward winning |
| **Your Pokemon faints** | -2 | Bad |
| **Double KO (momentum)** | +0.5 bonus | Pressing advantage |
| **Good Tera timing** | +1 | Tera that leads to KO or saves you |
| **Wasted Tera** | -1 | Tera that did nothing |
| **Damage dealt** | +small | Chip damage adds up |
| **Damage taken** | -small | Staying healthy |
| **Type advantage on field** | +0.2 | Good positioning |
| **Speed control** | +0.1 | Tailwind/Icy Wind value |

### Pokemon analogy:
Imagine if every time you played on ladder, you got a detailed score:
> "That Protect was good (+0.5), but you wasted Tera (-1), your double-up got 2 KOs (+4.5 with momentum bonus)"

The AI plays millions of games and optimizes for the highest score.

### What the AI learns:
- When copying experts fails, it finds better plays
- Risk/reward calculations (is this double-up worth it?)
- Tera timing optimization
- Reading common situations

---

## ⚔️ Stage 3: Real Ladder Experience (Self-Play)

### What happens:
The AI actually battles on **Pokemon Showdown** against real server conditions (though against copies of itself, not humans).

### The Population System

Instead of one AI, we train a whole "league" of AI players:

```
🏆 Hall of Fame (Best historical versions - never deleted)
     ↓ challenges
📊 Current Population (10-20 active AIs battling each other)
     ↓ promotes winners
🌱 New Challengers (Fresh versions created each round)
```

Each AI has an **ELO rating** (just like ladder):
- Start at 1500
- Win = gain ELO, Lose = lose ELO
- Best performers are saved forever

### Pokemon analogy:
It's like if you cloned yourself 20 times, had all the clones battle each other on ladder for a month, then kept only the best 5 clones for the next generation.

### What the AI learns:
- Adapting to different play styles
- Handling the unpredictability of real battles
- Punishing consistent strategies

---

## 🧬 Bonus: Team Evolution

Teams aren't fixed—they evolve alongside the AI!

### How it works:

1. Start with **20 random teams** from the valid VGC pool
2. Each team gets a **win rate** from actual battles
3. Every few rounds:
   - **Kill** the worst-performing teams
   - **Clone** the best teams with small changes ("mutations")
   - **Combine** good teams (take 3 Pokemon from Team A, 3 from Team B)

### Types of Mutations:
| Mutation | Example |
|----------|---------|
| Swap Pokemon | Replace Amoonguss → Rillaboom |
| Change item | Life Orb → Choice Specs |
| Change Tera type | Tera Fairy → Tera Ghost |
| Swap move | Moonblast → Dazzling Gleam |
| Adjust EVs | Move 32 EVs from SpA to Speed |

### Pokemon analogy:
It's natural selection for teams. Teams that win more "survive" and have "children" (variations). Bad teams go extinct.

---

## 🧠 How the AI "Sees" the Battle

The AI converts every battle state into **620 numbers** that describe:

### For each Pokemon (both players, 12 total):
- Species ID (one-hot encoded)
- Current HP %
- Status condition (burn, paralysis, etc.)
- Stat stages (-6 to +6 for each stat)
- Tera status (terastallized? what type?)
- Active or benched?
- Known moves

### Field conditions:
- Weather (Sun, Rain, Sand, Snow, None)
- Terrain (Grassy, Electric, Psychic, Misty, None)
- Trick Room active?
- Tailwind for each side?
- Entry hazards?

### Pokemon analogy:
Imagine if you had to describe the exact game state to someone over the phone using only numbers. That's what the AI works with.

---

## 🎯 How the AI Chooses Actions

Every turn in VGC, you make **2 decisions** (one per active Pokemon).

The AI considers **144 possible action combinations**:

For each slot (A and B):
- 4 moves × 2 (normal or with Tera) = 8 attack options
- 4 switch targets = 4 switch options
- Total = 12 options per slot
- Combined = 12 × 12 = 144 possible turn combinations

### Action Masking (Important!)
The AI knows which actions are **illegal**:
- Can't use a move with 0 PP
- Can't switch to a fainted Pokemon
- Can't Tera twice
- Can't target your own ally with single-target moves (usually)

Invalid actions are "masked out" so the AI never picks them.

---

## 📊 Current Features

### ✅ Working Now
| Feature | Description |
|---------|-------------|
| **Imitation Learning** | Learns from 21K+ expert replays |
| **PPO Training** | Self-improvement through simulation |
| **Self-Play Ladder** | Real Showdown battles against itself |
| **Team Evolution** | Genetic algorithm optimizes teams |
| **Action Masking** | Never picks illegal moves |
| **ELO Tracking** | Measures improvement over time |
| **Streamlit Dashboard** | Visual interface to watch AI |

### 🔮 Planned Features
| Feature | Description |
|---------|-------------|
| **Team Preview AI** | Learns which 4 Pokemon to bring |
| **Opponent Modeling** | Predicts what opponent will do |
| **Tera Timing Network** | Dedicated AI for Tera decisions |
| **Live Ladder Play** | Actually battle humans on ladder |

---

## 🎮 What Makes VGC Hard for AI?

### 1. Doubles Complexity
- 2 Pokemon acting simultaneously
- Spread moves, targeting, positioning
- 144 action combinations per turn (vs ~10 in singles)

### 2. Hidden Information
- Opponent's sets are unknown
- What moves do they have?
- What's their Tera type?
- Are they running Trick Room?

### 3. Team Preview Mind Games
- Picking the right 4 out of 6
- Predicting opponent's leads
- Planning switch-ins before the game starts

### 4. Tera Mechanic
- One-time resource, high impact
- Timing is crucial
- Wrong Tera can lose the game

### 5. Meta Adaptation
- Good strategies get countered
- Must constantly adapt
- Can't just memorize one approach

---

## 💡 Why This Approach?

### vs. Hard-coded bots:
Traditional Pokemon bots use rules like "use super-effective moves" or "switch when at low HP." These are predictable and can't handle complex situations.

### vs. Pure self-play:
Starting from scratch with self-play takes forever. By first learning from experts, the AI starts competent and then improves.

### vs. Tree search (like chess AI):
Pokemon has too much hidden information and randomness. You can't calculate 20 turns ahead when you don't know the opponent's moves or if Stone Edge will hit.

---

## 🏆 Goal

The ultimate goal is an AI that can:

1. ✅ Beat random/heuristic bots consistently
2. 🔄 Compete at ~1700+ ELO on Pokemon Showdown ladder
3. 🔮 Eventually challenge top human players

---

*Built with ❤️ for the VGC community*

