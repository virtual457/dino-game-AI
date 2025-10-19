<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<a id="readme-top"></a>

<!-- PROJECT TITLE -->
<div align="center">
  <h3 align="center">🎮 Dino Game Deep Reinforcement Learning</h3>
  <p align="center">
    An autonomous AI agent that learns to play Chrome's Dino game using Double Deep Q-Network (DDQN) with ResNet architecture. Features automated training loops, balanced experience replay, and real-time game interaction.
    <br/>
    <a href="https://github.com/virtual457/dino-game"><strong>Explore the docs »</strong></a>
    <br/><br/>
    <a href="https://github.com/virtual457/dino-game">View Demo</a>
    ·
    <a href="https://github.com/virtual457/dino-game/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/virtual457/dino-game/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## About The Project

This project implements a deep reinforcement learning system that autonomously learns to play Chrome's Dino game. The agent uses a ResNet-inspired Double DQN architecture with 1.5M parameters, processing 4-frame stacks for temporal velocity information and making real-time decisions at 16.67 FPS.

Key technical achievements:
- Automated training pipeline alternating between offline learning and online experience collection
- Balanced sampling strategy emphasizing crash scenarios (50/50 split)
- Equal penalty reward structure penalizing all frames in crash sequences
- Frame stacking for velocity information without explicit game state access
- Action persistence system eliminating flickering behaviors

This project serves as a portfolio piece demonstrating expertise in deep reinforcement learning, computer vision, and real-time system optimization.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Key Features

- ✅ Double DQN with ResNet architecture (1.5M parameters)
- ✅ Automated training loop with offline/online phases
- ✅ 4-frame stacking for temporal information
- ✅ Balanced experience replay (50/50 crash/alive sampling)
- ✅ Real-time screen capture and preprocessing
- ✅ Action persistence (no flickering)
- ✅ Hot-reloadable hyperparameters
- ✅ Comprehensive training visualization

## Architecture

```
Game Screen → Frame Capture → Preprocessing → Frame Stack (4 frames)
                                                      ↓
                                              ResNet DQN (1.5M)
                                                      ↓
                                          Q-values [Nothing, Jump, Duck]
                                                      ↓
                                              Action Selection
                                                      ↓
                                           Game Execution (16.67 FPS)
                                                      ↓
Experience Storage → Replay Buffer (FIFO, 8,500 cap) → Offline Training
                                                      ↓
                                              Model Checkpoint
                                                      ↓
                                           Repeat Collection
```

### Components

- **ResNet DQN**: 15 convolutional layers with skip connections, global average pooling
- **Automated Training Loop**: Alternates between offline training (until MSE convergence) and online collection (40 episodes)
- **Replay Buffer**: FIFO buffer with balanced sampling (16 negative + 16 positive experiences per batch)
- **Game Environment**: Screen capture, binary preprocessing, crash detection, action persistence

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- Windows OS (for Chrome automation)
- RTX GPU (recommended: RTX 4060 or better)
- 16GB RAM minimum

### Installation
```bash
# Clone the repository
git clone https://github.com/virtual457/dino-game.git
cd dino-game

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python mss pyautogui pygetwindow matplotlib

# Configure display settings in config.py
# Update CAPTURE_LEFT, CAPTURE_TOP for your monitor layout
```

### Quick Start
```bash
# Start automated training (recommended)
python automated_training_loop.py

# Or manually control training phases
python online_collector.py --num-episodes 40 --epsilon 0.2
python offline_trainer.py --max-steps 5000 --target-mse 9.0
```

### Configuration
Edit `config.py` to customize:
```python
TARGET_MSE_OFFLINE = 9.0      # Training convergence target
BUFFER_CAPACITY = 8500         # Experience replay size
LEARNING_RATE = 1e-4           # Adam optimizer learning rate
EPSILON_DECAY_RATE = 0.993     # Exploration decay rate
```

## Training Pipeline

### Automated Training Loop
1. **Offline Training Phase**: Train on replay buffer until MSE < target (max 5,000 steps)
2. **Online Collection Phase**: Play game for 40 episodes, collect experiences
3. **Repeat**: Cycle continues until Ctrl+C

### Reward Structure
- **Alive**: +5.0 for all actions (nothing, jump, duck)
- **Crash**: Last 8 frames ALL get -10.0 penalty
- **Philosophy**: Entire sequence contributes to crash, not single actions

### Training Speed
- **Offline**: ~7.6 training steps/second
- **Online**: 16.67 FPS gameplay
- **Memory**: ~5.4 GB RAM (pre-converted tensors)

## Model Architecture

### ResNet DQN (1.5M Parameters)
```
Input: (4, 84, 252) - 4 stacked binary frames
├── Conv Layer 1: 64 filters, 8×8 kernel, stride 4
├── Conv Layer 2: 128 filters, 4×4 kernel, stride 2
├── Conv Layer 3: 256 filters, 3×3 kernel, stride 1
├── Residual Block × 2 (256 channels)
├── Conv Layer 4: 512 filters, 3×3 kernel, stride 1
├── Residual Block × 2 (512 channels)
├── Conv Layer 5: 1024 filters, 3×3 kernel, stride 1
├── Residual Block × 1 (1024 channels)
├── Global Average Pooling: (1024, 3, 24) → (1024, 1, 1)
├── FC Layer 1: 1024 → 512 neurons
├── FC Layer 2: 512 → 256 neurons
└── FC Layer 3: 256 → 3 Q-values [Nothing, Jump, Duck]
```

## Roadmap

- [x] ResNet DQN architecture implementation
- [x] Frame stacking for temporal information
- [x] Double DQN algorithm
- [x] Automated training loop
- [x] Balanced experience replay
- [x] Equal penalty reward structure
- [x] Action persistence system
- [x] Hot-reloadable hyperparameters
- [ ] Multi-environment generalization
- [ ] Model compression for deployment
- [ ] Web-based training dashboard
- [ ] Advanced racing games (next project)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With

- **PyTorch** - Deep learning framework
- **OpenCV** - Image preprocessing
- **MSS** - Screen capture
- **PyAutoGUI** - Game control
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization

## Project Structure

```
dino-game/
├── automated_training_loop.py  # Main entry point
├── offline_trainer.py          # Train from replay buffer
├── online_collector.py         # Collect gameplay experiences
├── agent.py                    # DDQN agent implementation
├── model.py                    # ResNet DQN architecture
├── replay_buffer.py            # Experience replay (FIFO)
├── game_env.py                 # Chrome Dino environment
├── rewards.py                  # Reward calculation
├── config.py                   # Hyperparameters
├── episode_visualizer.py       # Episode visualization
├── visualizer.py               # Training progress plots
├── checkpoints/                # Model weights & buffer
└── utilities/                  # Debug and helper scripts
```

## Performance Metrics

### Current Benchmarks
- **Training Speed**: 7.6 steps/second (243 frames/second processed)
- **Memory Usage**: 5.4 GB RAM (pre-converted tensors), 3.5 GB GPU
- **Target MSE**: 9.0 (±3 error on Q-values ranging -80 to +500)
- **Episode Length**: Target 200-500 frames survival

### Technical Optimizations
- Pre-converted CPU tensors for fast sampling (80ms → 1ms)
- Balanced 50/50 sampling (crash emphasis)
- Batch size 32 for optimal GPU utilization
- FIFO buffer with 8,500 capacity (fits in 16GB RAM)

## Contributing

Contributions are welcome! Please open an issue to discuss changes or submit a PR following conventional guidelines.

Areas for contribution:
- Hyperparameter optimization
- Alternative reward structures
- Model architecture experiments
- Multi-game generalization
- Performance benchmarking

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE` for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Chandan Gowda K S – chandan.keelara@gmail.com

Project Link: https://github.com/virtual457/dino-game

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- Chrome Dino Game by Google
- PyTorch Team
- Deep Reinforcement Learning community
- OpenAI Gym documentation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/virtual457/dino-game.svg?style=for-the-badge
[forks-shield]: https://img.shields.io/github/forks/virtual457/dino-game.svg?style=for-the-badge
[stars-shield]: https://img.shields.io/github/stars/virtual457/dino-game.svg?style=for-the-badge
[issues-shield]: https://img.shields.io/github/issues/virtual457/dino-game.svg?style=for-the-badge
[license-shield]: https://img.shields.io/github/license/virtual457/dino-game.svg?style=for-the-badge
[contributors-url]: https://github.com/virtual457/dino-game/graphs/contributors
[forks-url]: https://github.com/virtual457/dino-game/network/members
[stars-url]: https://github.com/virtual457/dino-game/stargazers
[issues-url]: https://github.com/virtual457/dino-game/issues
[license-url]: https://github.com/virtual457/dino-game/blob/master/LICENSE
