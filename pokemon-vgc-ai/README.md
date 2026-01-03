# Pokemon VGC AI Battle Simulator and Team Optimizer

A reinforcement learning-based AI for Pokemon Video Game Championships (VGC) that learns from expert replays, improves through self-play, and battles on Pokemon Showdown.

## 🎯 Overview

This project implements a complete ML pipeline for training a competitive VGC battle AI:

```mermaid
flowchart LR
    subgraph Data["📊 Data"]
        D1[VGC-Bench<br/>21K+ battles]
        D2[Trajectory<br/>Parsing]
    end
    
    subgraph Training["🧠 Training Pipeline"]
        T1[Imitation<br/>Learning]
        T2[PPO<br/>Simulation]
        T3[Self-Play<br/>Real Battles]
    end
    
    subgraph Deployment["🚀 Deployment"]
        E1[Streamlit<br/>Dashboard]
        E2[Showdown<br/>Ladder]
    end
    
    D1 --> D2 --> T1 --> T2 --> T3 --> E1 & E2
```

---

## ✅ Current Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Data Pipeline** | ✅ Complete | 21K+ VGC battles parsed to trajectories |
| **Imitation Learning** | ✅ Complete | Behavioral cloning on expert replays |
| **PPO Training** | ✅ Complete | Simulated env (~800 FPS) with action masking |
| **Self-Play** | ✅ Complete | Real Showdown battles with ELO tracking |
| **Team Evolution** | ✅ Complete | Genetic algorithm for team optimization |
| **Streamlit Dashboard** | ✅ Complete | Battle visualization and analytics |
| **Tournament Mode** | ✅ Complete | Best-of-3 evaluation framework |

---

## 🏗️ Architecture

### Training Pipeline Architecture

```mermaid
flowchart TB
    subgraph Stage1["Stage 1: Imitation Learning"]
        IL1[Load VGC-Bench<br/>Trajectories]
        IL2[ImitationPolicy<br/>Network]
        IL3[Behavioral<br/>Cloning]
        IL1 --> IL3 --> IL2
    end
    
    subgraph Stage2["Stage 2: PPO Simulation"]
        PPO1[SimulatedVGCEnv<br/>~800 FPS]
        PPO2[MaskablePPO<br/>Action Masking]
        PPO3[Curriculum<br/>Learning]
        PPO1 --> PPO2
        PPO3 -.-> PPO2
    end
    
    subgraph Stage3["Stage 3: Self-Play"]
        SP1[Agent<br/>Population]
        SP2[Real Showdown<br/>Battles]
        SP3[ELO<br/>Tracking]
        SP4[Hall of<br/>Fame]
        SP1 --> SP2 --> SP3
        SP3 --> SP4
    end
    
    subgraph Stage4["Stage 4: Optional"]
        D1[Model<br/>Distillation]
        D2[Optimized<br/>Inference]
    end
    
    IL2 --> PPO2
    PPO2 --> SP1
    SP4 --> D1 --> D2
```

### Model Architecture

```mermaid
flowchart TB
    subgraph Input["Input (620 features)"]
        I1[Player Team<br/>6 × 50 features]
        I2[Opponent Team<br/>6 × 50 features]
        I3[Field State<br/>20 features]
    end
    
    subgraph Encoder["State Encoder"]
        E1[Pokemon<br/>Embeddings]
        E2[Team<br/>Attention]
        E3[Field<br/>Encoding]
    end
    
    subgraph Policy["Policy Network"]
        P1[Hidden Layers<br/>512 → 256]
        P2[Action Head<br/>144 outputs]
        P3[Value Head<br/>1 output]
    end
    
    subgraph Output["Output"]
        O1[Action<br/>Slot A: 12 options]
        O2[Action<br/>Slot B: 12 options]
    end
    
    I1 & I2 & I3 --> E1 & E2 & E3
    E1 & E2 & E3 --> P1
    P1 --> P2 & P3
    P2 --> O1 & O2
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd pokemon-vgc-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setup Pokemon Showdown

```bash
# Clone in parent directory
cd ..
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js

# Start server
node pokemon-showdown start --no-security
```

### 3. Run Training

```bash
cd pokemon-vgc-ai

# Minimal test (verify pipeline works)
python scripts/run_hybrid_training.py \
    --stages ppo_sim,self_play \
    --ppo-sim-timesteps 2000 \
    --self-play-iterations 2 \
    --self-play-real-battles

# Full training
python scripts/run_hybrid_training.py \
    --stages all \
    --imitation-epochs 20 \
    --ppo-sim-timesteps 100000 \
    --self-play-iterations 50 \
    --self-play-real-battles \
    --use-enhanced-selfplay \
    --evolve-teams
```

---

## 📜 Scripts Reference

### Main Training Script

| Script | Description |
|--------|-------------|
| `scripts/run_hybrid_training.py` | **Main training pipeline** - orchestrates all stages |

#### Usage Examples

```bash
# Stage 1 only: Imitation Learning
python scripts/run_hybrid_training.py --stages imitation --imitation-epochs 10

# Stage 2 only: PPO Simulation
python scripts/run_hybrid_training.py --stages ppo_sim --ppo-sim-timesteps 50000

# Stages 2-4: PPO + Self-Play with real battles
python scripts/run_hybrid_training.py \
    --stages ppo_sim,self_play \
    --self-play-real-battles \
    --use-enhanced-selfplay \
    --evolve-teams

# Full pipeline with all features
python scripts/run_hybrid_training.py \
    --stages all \
    --use-enhanced-selfplay \
    --self-play-real-battles \
    --evolve-teams \
    --mixed-precision \
    --distill
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stages` | `all` | Comma-separated: `imitation,ppo_sim,ppo_real,self_play,distill` |
| `--data-path` | `data/processed/trajectories_batched` | Path to trajectory data |
| `--output-dir` | `data/models/hybrid` | Output directory |
| `--continue-from` | None | Path to model to continue training |
| `--imitation-epochs` | 20 | Epochs for imitation learning |
| `--ppo-sim-timesteps` | 100000 | Timesteps for simulated PPO |
| `--ppo-real-timesteps` | 10000 | Timesteps for real Showdown PPO |
| `--self-play-iterations` | 20 | Self-play iterations |
| `--self-play-real-battles` | False | Use real Showdown battles |
| `--use-enhanced-selfplay` | False | Hall of Fame + League system |
| `--evolve-teams` | False | Enable team evolution |
| `--team-pool-size` | 20 | Teams in evolution pool |
| `--battle-timeout` | 60 | Timeout per real battle (seconds) |
| `--max-real-battles` | 5 | Max real battles per iteration |
| `--use-curriculum` | False | Progressive difficulty training |
| `--mixed-precision` | False | FP16 training (2x speedup) |
| `--distill` | False | Compress model after training |

### Utility Scripts

| Script | Description |
|--------|-------------|
| `scripts/start_showdown.sh` | Start Pokemon Showdown server |
| `scripts/run_full_pipeline.py` | Legacy full pipeline script |

---

## 📁 Project Structure

```
pokemon-vgc-ai/
├── config/                         # Hydra YAML configuration
│   ├── default.yaml                # Base config
│   ├── model/                      # Model architectures
│   ├── training/                   # Training presets
│   └── data/                       # Data sources
│
├── src/
│   ├── core/                       # Infrastructure
│   │   ├── config_schema.py        # Typed configurations
│   │   ├── container.py            # Dependency injection
│   │   └── plugins.py              # Plugin architecture
│   │
│   ├── data/                       # Data processing
│   │   ├── parsers/                # Battle log parsing
│   │   │   └── trajectory_builder.py  # Streaming processor
│   │   ├── storage/                # Parquet storage
│   │   ├── augmentation.py         # Data augmentation
│   │   ├── vgc_formats.py          # Multi-format support
│   │   └── monitoring.py           # Quality tracking
│   │
│   ├── engine/                     # Battle engine
│   │   ├── showdown/               # Showdown integration
│   │   └── state/                  # State encoding (620 features)
│   │
│   ├── ml/
│   │   ├── models/                 # Neural networks
│   │   │   ├── imitation_policy.py # Base policy
│   │   │   ├── team_preview.py     # 4-Pokemon selection AI
│   │   │   ├── opponent_model.py   # Action prediction
│   │   │   ├── tera_timing.py      # Tera decision network
│   │   │   └── optimized_inference.py  # <1ms inference
│   │   │
│   │   ├── training/               # Training modules
│   │   │   ├── imitation_learning.py
│   │   │   ├── rl_finetuning.py    # PPO with SimulatedVGCEnv
│   │   │   ├── self_play.py        # Basic self-play
│   │   │   ├── enhanced_self_play.py  # Hall of Fame + League
│   │   │   ├── reward_shaping.py   # Advanced rewards
│   │   │   ├── curriculum.py       # Progressive difficulty
│   │   │   ├── ladder_training.py  # Live ladder training
│   │   │   ├── distillation.py     # Model compression
│   │   │   ├── mixed_precision.py  # FP16 training
│   │   │   └── experiment_tracking.py
│   │   │
│   │   └── team_builder/           # Team optimization
│   │       ├── team.py             # Team/PokemonSet classes
│   │       ├── vgc_data.py         # 100+ Pokemon movesets
│   │       └── evolutionary.py     # Genetic algorithms
│   │
│   ├── eval/                       # Evaluation
│   │   ├── tournament.py           # Best-of-3 matches
│   │   ├── benchmark.py            # Automated testing
│   │   └── metrics.py              # ELO, win rate, etc.
│   │
│   └── app/                        # Streamlit UI
│       ├── main.py                 # Dashboard
│       └── components/
│           └── battle_viz.py       # Battle visualization
│
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
│
├── data/
│   ├── raw/vgc_bench/              # VGC-Bench dataset
│   ├── processed/trajectories_full/  # Parquet trajectories
│   └── models/                     # Model checkpoints
│
└── scripts/                        # Runnable scripts
```

---

## 🧠 Training Details

### Stage 1: Imitation Learning

```mermaid
sequenceDiagram
    participant D as VGC-Bench Data
    participant P as Parser
    participant T as Trainer
    participant M as ImitationPolicy
    
    D->>P: Raw battle logs (JSON)
    P->>P: Extract states & actions
    P->>T: Trajectories (Parquet)
    T->>M: Behavioral cloning
    T->>T: Cross-entropy loss
    T-->>M: Trained policy (~59% accuracy)
```

**Key Features:**
- Parses 21K+ expert VGC battles
- Outcome-weighted behavioral cloning
- Early stopping on validation accuracy

### Stage 2: PPO Simulation

```mermaid
sequenceDiagram
    participant E as SimulatedVGCEnv
    participant A as MaskablePPO
    participant R as RewardShaper
    
    loop Training Steps
        E->>A: State observation (620 dim)
        A->>A: Get action mask
        A->>E: Action (0-143)
        E->>R: Calculate shaped reward
        R-->>A: Reward signal
        A->>A: PPO update
    end
```

**Key Features:**
- ~800 FPS training speed
- Action masking prevents invalid moves
- Shaped rewards: KOs, HP differential, momentum

### Stage 3: Self-Play with Real Battles

```mermaid
sequenceDiagram
    participant P as Population
    participant S as Showdown Server
    participant E as ELO System
    participant H as Hall of Fame
    participant T as TeamPool
    
    loop Each Iteration
        P->>T: Request teams
        T-->>P: Assign teams
        P->>S: Battle request
        S->>S: Run real battle
        S-->>P: Battle result
        P->>E: Update ELO ratings
        E->>H: Check for Hall of Fame
        H->>T: Evolve winning teams
    end
```

**Key Features:**
- Real Pokemon Showdown battles
- Population-based training with ELO
- Hall of Fame preserves best agents
- Team evolution via genetic algorithms

---

## 🎮 Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run src/app/main.py
```

### Pages

| Page | Description |
|------|-------------|
| **Home** | Overview and quick stats |
| **Battle Arena** | Fight against AI with custom teams |
| **Team Builder** | Build and optimize teams |
| **Analytics** | Metagame stats and AI performance |
| **Settings** | Configuration and data management |

---

## 📊 Modules Integration Status

### ✅ Integrated in Training Pipeline

| Module | Location | Used In |
|--------|----------|---------|
| ImitationLearning | `training/imitation_learning.py` | Stage 1 |
| SimulatedVGCEnv | `training/rl_finetuning.py` | Stage 2 |
| MaskablePPO | `training/rl_finetuning.py` | Stage 2 |
| SelfPlayTrainer | `training/self_play.py` | Stage 4 |
| EnhancedSelfPlayTrainer | `training/enhanced_self_play.py` | Stage 4 |
| RewardShaper | `training/reward_shaping.py` | Stages 2-4 |
| TeamPoolManager | `training/enhanced_self_play.py` | Stage 4 |
| ExperimentTracker | `training/experiment_tracking.py` | All stages |
| Distillation | `training/distillation.py` | Stage 5 |

### ⏳ Standalone (Future Integration)

| Module | Location | Future Use |
|--------|----------|------------|
| TeamPreviewNetwork | `models/team_preview.py` | Before battles |
| OpponentPredictor | `models/opponent_model.py` | During action selection |
| TeraTiming | `models/tera_timing.py` | Tera decision support |
| FastInferenceEngine | `models/optimized_inference.py` | Production deployment |
| TournamentRunner | `eval/tournament.py` | Model evaluation |
| LadderTrainer | `training/ladder_training.py` | Live ladder training |

---

## 🔮 Future Integration Plans

```mermaid
flowchart TB
    subgraph Current["Current Pipeline"]
        C1[Imitation] --> C2[PPO Sim] --> C3[Self-Play]
    end
    
    subgraph Phase1["Phase 1: Strategic AI"]
        P1A[Team Preview AI]
        P1B[Opponent Modeling]
        P1C[Tera Timing]
    end
    
    subgraph Phase2["Phase 2: Production"]
        P2A[Optimized Inference]
        P2B[Live Ladder Training]
        P2C[Tournament Evaluation]
    end
    
    subgraph Phase3["Phase 3: Advanced"]
        P3A[Multi-Format Support]
        P3B[Metagame Adaptation]
        P3C[Team Building AI]
    end
    
    C3 --> P1A & P1B & P1C
    P1A & P1B & P1C --> P2A & P2B & P2C
    P2A & P2B & P2C --> P3A & P3B & P3C
```

### Phase 1: Strategic AI Integration
- [ ] Add TeamPreviewNetwork to battle initialization
- [ ] Integrate OpponentPredictor in action selection
- [ ] Use TeraTiming for Tera decisions

### Phase 2: Production Deployment
- [ ] Deploy FastInferenceEngine (<1ms per action)
- [ ] Enable LadderTrainer for live learning
- [ ] Automated tournament evaluation

### Phase 3: Advanced Features
- [ ] Support multiple VGC formats (Reg G, H, 2025)
- [ ] Dynamic metagame adaptation
- [ ] Full team building optimization

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific tests
pytest tests/unit/test_state_encoder.py -v
pytest tests/integration/test_training_pipeline.py -v
```

---

## 📈 Metrics

### Training Metrics

| Metric | Description |
|--------|-------------|
| **Win Rate** | % of battles won |
| **ELO** | Rating from matchup history (starts at 1500) |
| **FPS** | Training samples per second |
| **Action Accuracy** | % matching expert actions (imitation) |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **KO Efficiency** | KOs dealt per KO received |
| **Turn Efficiency** | Average turns per win |
| **Tera Usage** | Optimal Tera timing |
| **Switch Rate** | Positioning effectiveness |

---

## 📚 Data Sources

### VGC-Bench Dataset

| File | Format | Battles |
|------|--------|---------|
| `logs-gen9vgc2024regg.json` | VGC 2024 Reg G | 21,480 |
| `logs-gen9vgc2024regh.json` | VGC 2024 Reg H | ~30K |
| `logs-gen9vgc2025regg.json` | VGC 2025 Reg G | ~50K |

**Source**: [cameronangliss/vgc-battle-logs](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs)

---

## 🛠️ Requirements

- Python 3.11+
- Node.js 18+ (for Pokemon Showdown)
- ~4GB disk space (models + data)

### Key Dependencies

```
torch>=2.0
stable-baselines3>=2.0
sb3-contrib>=2.0
poke-env>=0.6
streamlit>=1.28
pandas
numpy
loguru
```

---

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Pokemon Showdown](https://pokemonshowdown.com/) - Battle simulator
- [poke-env](https://github.com/hsahovic/poke-env) - Python interface
- [VGC-Bench](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs) - Battle dataset
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
