"""Data augmentation for VGC battle trajectories.

Provides augmentation techniques to increase training data diversity:
- Perspective swap: Train from both players' viewpoints
- State noise: Add noise for robustness
- Action shuffling: Prevent action memorization
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from copy import deepcopy
from loguru import logger

# State dimension constants (from game_state.py)
STATE_DIM = 620
POKEMON_FEATURES = 50  # Per Pokemon with padding
NUM_POKEMON_PER_SIDE = 6
FIELD_FEATURES = 20


@dataclass
class Transition:
    """A single state transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    slot_a_target: Optional[int] = None
    slot_b_target: Optional[int] = None


@dataclass
class Trajectory:
    """A sequence of transitions from a battle."""
    battle_id: str
    player_perspective: str
    won: bool
    transitions: List[Transition]
    
    @property
    def length(self) -> int:
        return len(self.transitions)
    
    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)


class DataAugmenter:
    """Augments VGC battle trajectory data."""
    
    def __init__(
        self,
        perspective_swap: bool = True,
        add_noise: bool = True,
        noise_std: float = 0.01,
        shuffle_pokemon_order: bool = False,
    ):
        """Initialize augmenter.
        
        Args:
            perspective_swap: Enable perspective swapping
            add_noise: Add Gaussian noise to states
            noise_std: Standard deviation for noise
            shuffle_pokemon_order: Shuffle Pokemon order in state (experimental)
        """
        self.perspective_swap = perspective_swap
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.shuffle_pokemon_order = shuffle_pokemon_order
    
    def augment_trajectory(
        self, 
        trajectory: Trajectory
    ) -> List[Trajectory]:
        """Augment a single trajectory.
        
        Args:
            trajectory: Original trajectory
            
        Returns:
            List of augmented trajectories (including original)
        """
        augmented = [trajectory]
        
        # Perspective swap (if original is from one perspective, create the other)
        if self.perspective_swap:
            swapped = self._swap_perspective(trajectory)
            if swapped:
                augmented.append(swapped)
        
        # Add noisy versions
        if self.add_noise:
            noisy = self._add_noise(trajectory)
            augmented.append(noisy)
        
        return augmented
    
    def augment_batch(
        self,
        trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        """Augment a batch of trajectories.
        
        Args:
            trajectories: List of original trajectories
            
        Returns:
            List of all trajectories (original + augmented)
        """
        all_trajectories = []
        for traj in trajectories:
            augmented = self.augment_trajectory(traj)
            all_trajectories.extend(augmented)
        
        logger.info(f"Augmented {len(trajectories)} -> {len(all_trajectories)} trajectories")
        return all_trajectories
    
    def _swap_perspective(
        self, 
        trajectory: Trajectory
    ) -> Optional[Trajectory]:
        """Swap player/opponent perspective in trajectory.
        
        In the state representation:
        - First 6*50 = 300 features: Player's Pokemon
        - Next 6*50 = 300 features: Opponent's Pokemon
        - Last 20 features: Field state (some need to be swapped)
        
        Args:
            trajectory: Original trajectory
            
        Returns:
            Swapped trajectory or None if not possible
        """
        try:
            swapped_transitions = []
            
            for trans in trajectory.transitions:
                new_state = self._swap_state_perspective(trans.state)
                new_next_state = self._swap_state_perspective(trans.next_state)
                
                # Invert reward (winner becomes loser)
                new_reward = -trans.reward
                
                # Action remains the same conceptually but from opponent's view
                # For proper swap, we'd need to know opponent's action
                # For simplicity, keep the action (this is approximate)
                new_action = trans.action
                
                swapped_transitions.append(Transition(
                    state=new_state,
                    action=new_action,
                    reward=new_reward,
                    next_state=new_next_state,
                    done=trans.done,
                    slot_a_target=trans.slot_b_target,  # Swap targets
                    slot_b_target=trans.slot_a_target,
                ))
            
            return Trajectory(
                battle_id=f"{trajectory.battle_id}_swapped",
                player_perspective="opponent" if trajectory.player_perspective == "p1" else "p1",
                won=not trajectory.won,  # Inverse
                transitions=swapped_transitions,
            )
        except Exception as e:
            logger.warning(f"Failed to swap perspective: {e}")
            return None
    
    def _swap_state_perspective(self, state: np.ndarray) -> np.ndarray:
        """Swap player and opponent sections in state array.
        
        Args:
            state: Original state [620]
            
        Returns:
            Swapped state [620]
        """
        new_state = state.copy()
        
        # Swap player (0:300) and opponent (300:600) sections
        player_section = state[:300].copy()
        opponent_section = state[300:600].copy()
        
        new_state[:300] = opponent_section
        new_state[300:600] = player_section
        
        # Field state swaps (600:620)
        # Features like tailwind, reflect, light screen are side-specific
        # Indices (relative to 600):
        # 6: tailwind_player
        # 7: tailwind_opponent
        # 8: reflect_player
        # 9: light_screen_player
        # 10: reflect_opponent
        # 11: light_screen_opponent
        
        if len(state) > 606:
            # Swap tailwinds
            new_state[606], new_state[607] = state[607], state[606]
            # Swap screens
            new_state[608], new_state[610] = state[610], state[608]
            new_state[609], new_state[611] = state[611], state[609]
        
        return new_state
    
    def _add_noise(
        self, 
        trajectory: Trajectory,
    ) -> Trajectory:
        """Add Gaussian noise to state representations.
        
        Args:
            trajectory: Original trajectory
            
        Returns:
            Noisy trajectory
        """
        noisy_transitions = []
        
        for trans in trajectory.transitions:
            # Add noise to continuous features
            noisy_state = trans.state.copy()
            noisy_next_state = trans.next_state.copy()
            
            # Add noise to HP and stat values (not one-hot encodings)
            noise_state = np.random.normal(0, self.noise_std, noisy_state.shape)
            noise_next = np.random.normal(0, self.noise_std, noisy_next_state.shape)
            
            # Only add noise to non-binary features
            # HP fraction, stat boosts, etc. are in certain positions
            # For simplicity, add small noise everywhere (robust to all)
            noisy_state = noisy_state + noise_state * 0.5  # Reduced for binary features
            noisy_next_state = noisy_next_state + noise_next * 0.5
            
            noisy_transitions.append(Transition(
                state=noisy_state.astype(np.float32),
                action=trans.action,
                reward=trans.reward,  # Don't add noise to rewards
                next_state=noisy_next_state.astype(np.float32),
                done=trans.done,
                slot_a_target=trans.slot_a_target,
                slot_b_target=trans.slot_b_target,
            ))
        
        return Trajectory(
            battle_id=f"{trajectory.battle_id}_noisy",
            player_perspective=trajectory.player_perspective,
            won=trajectory.won,
            transitions=noisy_transitions,
        )
    
    def augment_state(
        self,
        state: np.ndarray,
        noise: bool = True,
    ) -> np.ndarray:
        """Augment a single state (for online augmentation).
        
        Args:
            state: State vector
            noise: Whether to add noise
            
        Returns:
            Augmented state
        """
        aug_state = state.copy()
        
        if noise:
            noise_vec = np.random.normal(0, self.noise_std, state.shape)
            aug_state = aug_state + noise_vec
        
        return aug_state.astype(np.float32)


def create_augmenter(
    perspective_swap: bool = True,
    noise: bool = True,
    noise_std: float = 0.01,
) -> DataAugmenter:
    """Factory function to create a data augmenter.
    
    Args:
        perspective_swap: Enable perspective swapping
        noise: Enable noise augmentation
        noise_std: Noise standard deviation
        
    Returns:
        Configured DataAugmenter
    """
    return DataAugmenter(
        perspective_swap=perspective_swap,
        add_noise=noise,
        noise_std=noise_std,
    )


def augment_parquet_batch(
    input_path: str,
    output_path: str,
    perspective_swap: bool = True,
    add_noise: bool = True,
) -> int:
    """Augment a batch of trajectories from Parquet file.
    
    Args:
        input_path: Path to input Parquet file
        output_path: Path to output Parquet file
        perspective_swap: Enable perspective swapping
        add_noise: Enable noise augmentation
        
    Returns:
        Number of augmented transitions
    """
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    # Read input
    table = pq.read_table(input_path)
    df = table.to_pandas()
    
    logger.info(f"Loaded {len(df)} transitions from {input_path}")
    
    augmenter = DataAugmenter(
        perspective_swap=perspective_swap,
        add_noise=add_noise,
    )
    
    # Process each row
    augmented_rows = []
    
    for _, row in df.iterrows():
        # Original row
        augmented_rows.append(row.to_dict())
        
        # Perspective swap
        if perspective_swap:
            swapped = row.to_dict()
            state = np.frombuffer(row['state'], dtype=np.float32)
            next_state = np.frombuffer(row['next_state'], dtype=np.float32)
            
            swapped['state'] = augmenter._swap_state_perspective(state).tobytes()
            swapped['next_state'] = augmenter._swap_state_perspective(next_state).tobytes()
            swapped['reward'] = -row['reward']
            swapped['won'] = not row['won']
            swapped['battle_id'] = f"{row['battle_id']}_swapped"
            augmented_rows.append(swapped)
        
        # Noisy version
        if add_noise:
            noisy = row.to_dict()
            state = np.frombuffer(row['state'], dtype=np.float32)
            next_state = np.frombuffer(row['next_state'], dtype=np.float32)
            
            noisy['state'] = augmenter.augment_state(state).tobytes()
            noisy['next_state'] = augmenter.augment_state(next_state).tobytes()
            noisy['battle_id'] = f"{row['battle_id']}_noisy"
            augmented_rows.append(noisy)
    
    # Save augmented data
    output_table = pa.Table.from_pylist(augmented_rows)
    pq.write_table(output_table, output_path, compression='snappy')
    
    logger.info(f"Saved {len(augmented_rows)} augmented transitions to {output_path}")
    return len(augmented_rows)

