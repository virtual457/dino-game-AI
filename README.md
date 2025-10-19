# 🦖 Chrome Dino Game - Deep Q-Network Agent

A reinforcement learning agent that learns to play the Chrome Dino game using Double DQN.

---

## 🎮 What Does It Do?

Trains an AI agent to:
- Jump over cacti 🌵
- Duck under birds 🦅
- Survive as long as possible in the Chrome Dino game

---

## 🚀 Quick Start

### 1. Start Training
```bash
python main.py
```

### 2. Debug & Analyze
```bash
# Run complete diagnostics
python run_diagnostics.py

# Or run individual tools
python diagnostics.py        # System check
python analyze_training.py   # Model analysis
python quick_test.py         # Performance test
python debug_dashboard.py    # Live visual debugging
```

### 3. Apply Fixes (if needed)
```bash
python quick_fixes.py [fix_name]

# Examples:
python quick_fixes.py reset_epsilon      # More exploration
python quick_fixes.py increase_rewards   # Value survival more
python quick_fixes.py reset_training     # Start fresh
```

---

## 📋 Requirements

```bash
pip install torch torchvision
pip install numpy opencv-python matplotlib
pip install mss pygetwindow pyautogui
```

**GPU Recommended:** Install PyTorch with CUDA for faster training
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🏗️ Architecture

**Algorithm:** Double DQN (fixes overestimation bias)

**Input:** 
- 4 stacked frames (84x252 pixels each)
- Provides velocity information

**Actions:**
- 0: Do nothing
- 1: Jump
- 2: Duck

**Network:**
```
4 Frames → CNN (3 layers) → Flatten → FC (512→256→3) → Q-values
```

**Key Features:**
- ✅ Frame stacking for velocity
- ✅ Double DQN (prevents overestimation)
- ✅ Experience replay (100k buffer)
- ✅ Reward shaping (diminishing penalties near death)
- ✅ 10 FPS locked training for consistency

---

## 📊 Current Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Frame Stack | 4 frames | Velocity detection |
| State Size | 84×252 | Preprocessed game frames |
| Buffer Size | 100,000 | Experience replay |
| Learning Rate | 1e-4 | Stable pixel-based learning |
| Gamma | 0.99 | Long-term survival focus |
| Epsilon Decay | 0.9995 | Exponential exploration decay |
| Batch Size | 32 | Standard mini-batch |
| Target Update | Every 10k steps | Network stability |

---

## 📈 Training Progress

Training auto-saves every 5,000 steps and generates plots every 10 episodes.

**Monitor:**
- Episode length (should increase)
- Average reward (should increase)
- Action distribution (should be balanced)
- Epsilon (should decay from 1.0 → 0.01)

---

## 🐛 Debugging

**See [DEBUGGING_TOOLKIT.md](DEBUGGING_TOOLKIT.md) for complete guide**

### Quick Diagnostics:
```bash
python run_diagnostics.py
```

### Common Issues:

| Issue | Quick Fix |
|-------|-----------|
| Agent does nothing | `python quick_fixes.py reset_epsilon` |
| Dying immediately | Train longer, check preprocessing |
| Q-values exploding | `python quick_fixes.py adjust_lr_low` |
| Was good, now bad | `python quick_fixes.py restore_backup` |

---

## 📁 Project Structure

```
dino-game/
├── main.py                    # Training entry point
├── trainer.py                 # Training loop
├── agent.py                   # DDQN agent
├── model.py                   # Neural network
├── game_env.py                # Game environment
├── config.py                  # Configuration
│
├── Debugging Tools
│   ├── diagnostics.py         # System health
│   ├── analyze_training.py    # Model analysis
│   ├── debug_dashboard.py     # Live debugging
│   ├── quick_test.py          # Performance test
│   ├── quick_fixes.py         # Automated fixes
│   └── run_diagnostics.py     # Full workflow
│
├── Documentation
│   ├── README.md              # This file
│   ├── DEBUGGING_TOOLKIT.md   # Complete debugging guide
│   └── DEBUG_GUIDE.md         # Detailed solutions
│
└── checkpoints/               # Saved models & plots
```

---

## 🎯 Usage Examples

### Basic Training
```bash
# Start training
python main.py

# Press Ctrl+C to stop (saves automatically)
```

### Monitor Progress
```bash
# Quick performance check
python quick_test.py

# Visual debugging
python debug_dashboard.py
```

### Troubleshooting
```bash
# Full diagnostics
python run_diagnostics.py

# Apply fix
python quick_fixes.py reset_epsilon

# Continue training
python main.py
```

---

## 🎓 Understanding the Agent

### Frame Preprocessing
1. Capture raw frame (950×300)
2. Convert to grayscale
3. Binary threshold (black/white)
4. Resize to 84×252
5. Normalize to [0, 1]

### Learning Process
1. **Observe**: Capture 4 frames (velocity)
2. **Decide**: Network predicts Q-values for each action
3. **Act**: Execute action (jump/duck/nothing)
4. **Learn**: Store experience, train on mini-batch
5. **Update**: Sync target network periodically

### Reward Structure
- **Alive**: +0.1 per frame
- **Death**: -10.0 on crash
- **Shaped**: Diminishing penalties (3 frames before crash)

---

## 📊 Performance Metrics

### Training Stages

| Stage | Frames/Episode | Status |
|-------|---------------|---------|
| Random | 50-100 | Not learned yet |
| Basic | 100-300 | Learning obstacles |
| Intermediate | 300-1000 | Consistent jumping |
| Advanced | 1000-5000 | Handling birds |
| Master | 5000+ | Indefinite survival |

---

## 🔧 Hyperparameter Tuning

### To Adjust Performance:

**More Exploration:**
```python
# config.py
EPSILON_DECAY_RATE = 0.999  # Slower decay
```

**Faster Learning:**
```python
LEARNING_RATE = 5e-4  # Higher LR
BATCH_SIZE = 64       # Larger batches
```

**Better Memory:**
```python
BUFFER_CAPACITY = 500000  # More experiences
```

**Stronger Rewards:**
```python
REWARD_ALIVE = 0.5   # Higher survival value
REWARD_DEATH = -20.0 # Stronger death penalty
```

---

## 🎥 What to Watch For

### Good Signs ✅
- Episode length increasing
- Epsilon decaying smoothly (1.0 → 0.01)
- Q-values stable (-10 to +10)
- Balanced action usage

### Warning Signs ⚠️
- Episode length stuck <100
- One action used 90%+
- Q-values exploding (>100)
- Loss increasing

---

## 🚀 Next Steps

1. **Run diagnostics**: `python run_diagnostics.py`
2. **Check performance**: `python quick_test.py`
3. **Apply fixes if needed**: `python quick_fixes.py [fix]`
4. **Continue training**: `python main.py`
5. **Iterate** until agent masters the game!

---

## 📚 Resources

- **DEBUG_GUIDE.md** - Detailed problem-solving guide
- **DEBUGGING_TOOLKIT.md** - Tool documentation
- **config.py** - All hyperparameters
- **checkpoints/** - Saved models and plots

---

## 💡 Tips

1. **GPU Required**: Training is slow on CPU
2. **Monitor Early**: Check performance every 10k steps
3. **Backup Often**: System auto-saves, but keep backups
4. **Visual Debug**: Watch `debug_dashboard.py` to understand behavior
5. **Patience**: Good performance needs 50k-100k steps

---

## 🎯 Goal

**Target Performance:** 1000+ frames per episode consistently

**Current Status:** Check with `python quick_test.py`

---

**Ready to train? Run `python main.py` and let's make this dino unstoppable! 🦖💪**

For debugging help: `python run_diagnostics.py`
