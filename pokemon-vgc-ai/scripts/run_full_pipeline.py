#!/usr/bin/env python3
"""Full training pipeline for Pokemon VGC AI.

This script orchestrates the complete training process:
1. Process all battle replays into trajectories
2. Convert to Parquet format
3. Train imitation learning model
4. Fine-tune with PPO and curriculum learning
5. Run self-play improvement
6. Benchmark and generate report

Usage:
    python scripts/run_full_pipeline.py --all
    python scripts/run_full_pipeline.py --stage data
    python scripts/run_full_pipeline.py --stage imitation
    python scripts/run_full_pipeline.py --stage ppo
    python scripts/run_full_pipeline.py --stage selfplay
    python scripts/run_full_pipeline.py --stage benchmark
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


@dataclass
class PipelineConfig:
    """Configuration for the full training pipeline."""
    
    # Data paths
    raw_data_path: Path = Path("data/raw/vgc_bench/logs-gen9vgc2024regg.json")
    trajectory_dir: Path = Path("data/processed/trajectories")
    
    # Processing
    max_battles: Optional[int] = None  # None = process all
    batch_size: int = 5000  # Battles per batch for memory efficiency
    output_format: str = "parquet"  # "json" or "parquet"
    
    # Imitation learning
    imitation_epochs: int = 20
    imitation_batch_size: int = 512
    imitation_lr: float = 1e-3
    model_type: str = "enhanced"  # "simple", "attention", or "enhanced"
    imitation_save_dir: Path = Path("data/models/imitation")
    
    # RL fine-tuning
    ppo_timesteps: int = 200_000
    use_curriculum: bool = True
    ppo_save_dir: Path = Path("data/models/ppo_finetuned")
    
    # Self-play
    selfplay_iterations: int = 20
    selfplay_population_size: int = 10
    selfplay_save_dir: Path = Path("data/models/self_play")
    
    # Benchmark
    benchmark_episodes: int = 100
    report_dir: Path = Path("data/reports")


class PipelineRunner:
    """Orchestrates the full training pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.results: Dict[str, Any] = {}
        self.start_time = None
    
    def run_all(self):
        """Run complete pipeline."""
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("STARTING FULL VGC AI TRAINING PIPELINE")
        logger.info("=" * 60)
        
        try:
            self.run_data_processing()
            self.run_imitation_learning()
            self.run_ppo_finetuning()
            self.run_self_play()
            self.run_benchmark()
            self.generate_report()
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            elapsed = time.time() - self.start_time
            logger.info(f"Total pipeline time: {elapsed/60:.1f} minutes")
    
    def run_data_processing(self):
        """Stage 1: Process battle replays into trajectories."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: DATA PROCESSING")
        logger.info("=" * 60)
        
        stage_start = time.time()
        
        from src.data.parsers.trajectory_builder import process_vgc_bench_file
        from src.data.storage.parquet_storage import convert_json_to_parquet
        
        input_path = self.config.raw_data_path
        output_dir = self.config.trajectory_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if data exists
        if not input_path.exists():
            logger.error(f"Raw data not found: {input_path}")
            logger.info("Please download VGC-Bench data first")
            raise FileNotFoundError(f"Missing: {input_path}")
        
        # Check how many battles
        logger.info(f"Loading battle data from {input_path}...")
        with open(input_path, 'r') as f:
            import json as json_lib
            data = json_lib.load(f)
        
        total_battles = len(data)
        del data  # Free memory
        
        logger.info(f"Found {total_battles} battles to process")
        
        # Process all battles
        max_battles = self.config.max_battles or total_battles
        logger.info(f"Processing {max_battles} battles...")
        
        num_trajectories = process_vgc_bench_file(
            input_path,
            output_dir,
            max_battles=max_battles,
        )
        
        self.results['data'] = {
            'total_battles': total_battles,
            'processed_battles': min(max_battles, total_battles),
            'trajectories': num_trajectories,
        }
        
        # Convert to Parquet if requested
        json_file = output_dir / f"trajectories_{input_path.stem}.json"
        
        if self.config.output_format == "parquet" and json_file.exists():
            logger.info("Converting to Parquet format...")
            parquet_path, stats = convert_json_to_parquet(json_file)
            
            self.results['data']['parquet_path'] = str(parquet_path)
            self.results['data']['compression_ratio'] = stats.get('compression_ratio', 1.0)
            
            logger.info(f"Saved Parquet file: {parquet_path}")
        
        elapsed = time.time() - stage_start
        self.results['data']['time_seconds'] = elapsed
        logger.info(f"Data processing completed in {elapsed:.1f} seconds")
    
    def run_imitation_learning(self):
        """Stage 2: Train imitation learning model."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: IMITATION LEARNING")
        logger.info("=" * 60)
        
        stage_start = time.time()
        
        from src.ml.training.imitation_learning import TrainingConfig, ImitationTrainer
        
        # Find trajectory file
        traj_dir = self.config.trajectory_dir
        
        # Prefer Parquet if available
        parquet_files = list(traj_dir.glob("*.parquet"))
        json_files = list(traj_dir.glob("*.json"))
        
        if parquet_files:
            trajectory_path = parquet_files[0]
            data_format = "parquet"
        elif json_files:
            trajectory_path = json_files[0]
            data_format = "json"
        else:
            raise FileNotFoundError(f"No trajectory files found in {traj_dir}")
        
        logger.info(f"Using trajectory file: {trajectory_path}")
        
        # Configure training
        config = TrainingConfig(
            trajectory_path=str(trajectory_path),
            data_format=data_format,
            epochs=self.config.imitation_epochs,
            batch_size=self.config.imitation_batch_size,
            learning_rate=self.config.imitation_lr,
            model_type=self.config.model_type,
            save_dir=str(self.config.imitation_save_dir),
            use_amp=True,
        )
        
        # Train
        trainer = ImitationTrainer(config)
        best_model_path = trainer.train()
        
        self.results['imitation'] = {
            'model_path': str(best_model_path),
            'best_val_loss': trainer.best_val_loss,
            'model_type': self.config.model_type,
        }
        
        elapsed = time.time() - stage_start
        self.results['imitation']['time_seconds'] = elapsed
        logger.info(f"Imitation learning completed in {elapsed/60:.1f} minutes")
    
    def run_ppo_finetuning(self):
        """Stage 3: Fine-tune with PPO."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: PPO FINE-TUNING")
        logger.info("=" * 60)
        
        stage_start = time.time()
        
        from src.ml.training.rl_finetuning import (
            RLConfig, train_with_curriculum, train_ppo_finetuned
        )
        
        # Find imitation model
        imitation_model = self.config.imitation_save_dir / "best_model.pt"
        if not imitation_model.exists():
            logger.warning(f"Imitation model not found at {imitation_model}")
            imitation_model = None
        
        # Configure
        config = RLConfig(
            imitation_model_path=str(imitation_model) if imitation_model else "",
            total_timesteps=self.config.ppo_timesteps,
            use_curriculum=self.config.use_curriculum,
            save_dir=str(self.config.ppo_save_dir),
        )
        
        # Train
        if self.config.use_curriculum:
            model_path = train_with_curriculum(config)
        else:
            model_path = train_ppo_finetuned(config)
        
        self.results['ppo'] = {
            'model_path': str(model_path),
            'timesteps': self.config.ppo_timesteps,
            'curriculum': self.config.use_curriculum,
        }
        
        elapsed = time.time() - stage_start
        self.results['ppo']['time_seconds'] = elapsed
        logger.info(f"PPO fine-tuning completed in {elapsed/60:.1f} minutes")
    
    def run_self_play(self):
        """Stage 4: Self-play improvement."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4: SELF-PLAY IMPROVEMENT")
        logger.info("=" * 60)
        
        stage_start = time.time()
        
        from src.ml.training.self_play import SelfPlayTrainer, SelfPlayConfig
        
        # Configure
        config = SelfPlayConfig(
            population_size=self.config.selfplay_population_size,
            num_iterations=self.config.selfplay_iterations,
            save_dir=str(self.config.selfplay_save_dir),
        )
        
        # Train
        trainer = SelfPlayTrainer(config)
        
        for i in range(self.config.selfplay_iterations):
            stats = trainer.run_iteration()
            if i % 5 == 0:
                logger.info(f"Iteration {i+1}: Best ELO = {stats.get('best_elo', 'N/A')}")
        
        self.results['selfplay'] = {
            'iterations': self.config.selfplay_iterations,
            'population_size': self.config.selfplay_population_size,
            'final_best_elo': trainer.get_best_agent().elo if hasattr(trainer, 'get_best_agent') else None,
        }
        
        elapsed = time.time() - stage_start
        self.results['selfplay']['time_seconds'] = elapsed
        logger.info(f"Self-play completed in {elapsed/60:.1f} minutes")
    
    def run_benchmark(self):
        """Stage 5: Run benchmarks."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 5: BENCHMARKING")
        logger.info("=" * 60)
        
        stage_start = time.time()
        
        from src.ml.training.rl_finetuning import evaluate_model
        from src.eval.benchmark import BenchmarkSuite
        
        # Find best model
        ppo_model = self.config.ppo_save_dir / "ppo_curriculum.zip"
        if not ppo_model.exists():
            ppo_model = self.config.ppo_save_dir / "ppo_finetuned.zip"
        
        if ppo_model.exists():
            metrics = evaluate_model(ppo_model, n_episodes=self.config.benchmark_episodes)
            
            self.results['benchmark'] = {
                'model': str(ppo_model),
                'episodes': self.config.benchmark_episodes,
                **metrics,
            }
        else:
            logger.warning("No trained PPO model found for benchmarking")
            self.results['benchmark'] = {'error': 'No model found'}
        
        elapsed = time.time() - stage_start
        self.results['benchmark']['time_seconds'] = elapsed
        logger.info(f"Benchmarking completed in {elapsed:.1f} seconds")
    
    def generate_report(self):
        """Generate final training report."""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING FINAL REPORT")
        logger.info("=" * 60)
        
        report_dir = self.config.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"training_report_{timestamp}.json"
        
        # Add summary
        total_time = sum(
            r.get('time_seconds', 0) 
            for r in self.results.values() 
            if isinstance(r, dict)
        )
        
        report = {
            'timestamp': timestamp,
            'total_time_minutes': total_time / 60,
            'config': {
                'model_type': self.config.model_type,
                'imitation_epochs': self.config.imitation_epochs,
                'ppo_timesteps': self.config.ppo_timesteps,
                'selfplay_iterations': self.config.selfplay_iterations,
            },
            'stages': self.results,
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        
        if 'data' in self.results:
            logger.info(f"Data: {self.results['data'].get('trajectories', 'N/A')} trajectories")
        
        if 'imitation' in self.results:
            logger.info(f"Imitation: best_val_loss = {self.results['imitation'].get('best_val_loss', 'N/A'):.4f}")
        
        if 'benchmark' in self.results:
            logger.info(f"Win rate: {self.results['benchmark'].get('win_rate', 'N/A'):.2%}")
            logger.info(f"Mean reward: {self.results['benchmark'].get('mean_reward', 'N/A'):.2f}")
        
        logger.info(f"Total time: {total_time/60:.1f} minutes")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run VGC AI training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run complete pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["data", "imitation", "ppo", "selfplay", "benchmark"],
        help="Run specific stage"
    )
    parser.add_argument(
        "--max-battles",
        type=int,
        default=None,
        help="Max battles to process (None = all)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="enhanced",
        choices=["simple", "attention", "enhanced"],
        help="Model architecture"
    )
    parser.add_argument(
        "--imitation-epochs",
        type=int,
        default=20,
        help="Imitation learning epochs"
    )
    parser.add_argument(
        "--ppo-timesteps",
        type=int,
        default=200_000,
        help="PPO training timesteps"
    )
    parser.add_argument(
        "--selfplay-iterations",
        type=int,
        default=20,
        help="Self-play iterations"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("data/logs/pipeline_{time}.log", rotation="100 MB")
    
    # Build config
    config = PipelineConfig(
        max_battles=args.max_battles,
        model_type=args.model_type,
        imitation_epochs=args.imitation_epochs,
        ppo_timesteps=args.ppo_timesteps,
        selfplay_iterations=args.selfplay_iterations,
    )
    
    runner = PipelineRunner(config)
    
    if args.all:
        runner.run_all()
    elif args.stage == "data":
        runner.run_data_processing()
    elif args.stage == "imitation":
        runner.run_imitation_learning()
    elif args.stage == "ppo":
        runner.run_ppo_finetuning()
    elif args.stage == "selfplay":
        runner.run_self_play()
    elif args.stage == "benchmark":
        runner.run_benchmark()
    else:
        parser.print_help()
        logger.info("\nRun with --all for complete pipeline or --stage for specific stage")


if __name__ == "__main__":
    main()

