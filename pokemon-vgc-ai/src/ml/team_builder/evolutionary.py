"""Evolutionary algorithm for VGC team optimization."""

import random
import copy
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from loguru import logger

from src.config import config, MODELS_DIR
from src.ml.team_builder.team import (
    Team, PokemonSet, EVSpread,
    VGC_POKEMON_POOL, VGC_ITEMS, TERA_TYPES,
)


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm."""
    
    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 5
    elite_size: int = 10
    stagnation_limit: int = 50


class TeamEvolver:
    """Evolutionary algorithm for optimizing VGC teams."""
    
    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        fitness_fn: Optional[Callable[[Team], float]] = None,
        save_dir: Optional[Path] = None,
    ):
        """Initialize the evolver.
        
        Args:
            config: Evolution configuration
            fitness_fn: Custom fitness function (team -> score)
            save_dir: Directory to save results
        """
        self.config = config or EvolutionConfig()
        self.fitness_fn = fitness_fn or self._default_fitness
        self.save_dir = save_dir or (MODELS_DIR / "team_evolution")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.population: List[Team] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.best_team: Optional[Team] = None
    
    def _default_fitness(self, team: Team) -> float:
        """Default fitness function based on team composition heuristics.
        
        A more sophisticated version would use the battle AI to evaluate.
        """
        score = 0.0
        
        # Validity check (penalize invalid teams)
        if not team.is_valid():
            return -100.0
        
        # Diversity bonus (different types, roles)
        species_types = set()
        for p in team.pokemon:
            # Simple type inference from species name
            species_types.add(p.species.lower()[:3])
        score += len(species_types) * 5
        
        # Item synergy (penalize duplicate items already handled)
        score += 10  # Base score for valid item clause
        
        # EV spread efficiency
        for p in team.pokemon:
            ev_total = sum(p.evs.to_list())
            if 500 <= ev_total <= 510:
                score += 2
        
        # Random factor to simulate battle performance
        # In real implementation, this would be actual battle results
        score += random.gauss(50, 10)
        
        return score
    
    def initialize_population(self):
        """Create initial random population."""
        logger.info(f"Initializing population of {self.config.population_size} teams...")
        
        self.population = []
        for i in range(self.config.population_size):
            team = Team.random()
            team.name = f"Team_{i}"
            self.population.append(team)
        
        logger.info("Population initialized")
    
    def evaluate_population(self) -> List[Tuple[Team, float]]:
        """Evaluate fitness of all teams.
        
        Returns:
            List of (team, fitness) tuples sorted by fitness
        """
        evaluated = []
        for team in self.population:
            fitness = self.fitness_fn(team)
            evaluated.append((team, fitness))
        
        # Sort by fitness descending
        evaluated.sort(key=lambda x: x[1], reverse=True)
        return evaluated
    
    def select_parent(
        self, 
        evaluated: List[Tuple[Team, float]]
    ) -> Team:
        """Select parent using tournament selection.
        
        Args:
            evaluated: List of (team, fitness) tuples
            
        Returns:
            Selected parent team
        """
        # Tournament selection
        tournament = random.sample(evaluated, self.config.tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]
    
    def crossover(self, parent1: Team, parent2: Team) -> Team:
        """Create offspring by combining two parent teams.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child team
        """
        child_pokemon = []
        
        # Take 3 Pokemon from each parent
        p1_pokemon = random.sample(parent1.pokemon, 3)
        p2_pokemon = random.sample(parent2.pokemon, 3)
        
        # Combine and ensure no duplicate species
        used_species = set()
        
        for p in p1_pokemon + p2_pokemon:
            if p.species not in used_species:
                child_pokemon.append(copy.deepcopy(p))
                used_species.add(p.species)
        
        # Fill remaining slots if needed
        while len(child_pokemon) < 6:
            species = random.choice(VGC_POKEMON_POOL)
            if species not in used_species:
                child_pokemon.append(PokemonSet.random(species))
                used_species.add(species)
        
        # Fix item clause violations
        used_items = set()
        for p in child_pokemon:
            while p.item in used_items:
                p.item = random.choice(VGC_ITEMS)
            used_items.add(p.item)
        
        return Team(pokemon=child_pokemon[:6])
    
    def mutate(self, team: Team) -> Team:
        """Apply mutations to a team.
        
        Args:
            team: Team to mutate
            
        Returns:
            Mutated team (may be same object)
        """
        mutated = copy.deepcopy(team)
        
        for pokemon in mutated.pokemon:
            # Mutate each aspect with some probability
            
            # Species mutation (swap with pool)
            if random.random() < self.config.mutation_rate * 0.5:
                new_species = random.choice(VGC_POKEMON_POOL)
                # Check for species clause
                current_species = [p.species for p in mutated.pokemon]
                if new_species not in current_species:
                    pokemon.species = new_species
            
            # Item mutation
            if random.random() < self.config.mutation_rate:
                current_items = [p.item for p in mutated.pokemon if p.item != pokemon.item]
                new_item = random.choice(VGC_ITEMS)
                while new_item in current_items:
                    new_item = random.choice(VGC_ITEMS)
                pokemon.item = new_item
            
            # Tera type mutation
            if random.random() < self.config.mutation_rate:
                pokemon.tera_type = random.choice(TERA_TYPES)
            
            # EV mutation (small adjustments)
            if random.random() < self.config.mutation_rate:
                pokemon.evs = EVSpread.random()
        
        return mutated
    
    def evolve_generation(self) -> Tuple[Team, float]:
        """Evolve one generation.
        
        Returns:
            Tuple of (best_team, best_fitness) for this generation
        """
        # Evaluate current population
        evaluated = self.evaluate_population()
        
        best_team, best_fitness = evaluated[0]
        self.best_fitness_history.append(best_fitness)
        
        if self.best_team is None or best_fitness > self.fitness_fn(self.best_team):
            self.best_team = copy.deepcopy(best_team)
        
        # Create new population
        new_population = []
        
        # Elitism: keep best teams
        for team, _ in evaluated[:self.config.elite_size]:
            new_population.append(copy.deepcopy(team))
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = self.select_parent(evaluated)
                parent2 = self.select_parent(evaluated)
                child = self.crossover(parent1, parent2)
            else:
                # Clone parent
                parent = self.select_parent(evaluated)
                child = copy.deepcopy(parent)
            
            # Mutation
            child = self.mutate(child)
            child.name = f"Team_g{self.generation}_{len(new_population)}"
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return best_team, best_fitness
    
    def run(
        self, 
        generations: Optional[int] = None,
        verbose: bool = True,
    ) -> Team:
        """Run the evolutionary algorithm.
        
        Args:
            generations: Number of generations (uses config if None)
            verbose: Whether to log progress
            
        Returns:
            Best team found
        """
        generations = generations or self.config.generations
        
        if not self.population:
            self.initialize_population()
        
        logger.info(f"Starting evolution for {generations} generations...")
        
        stagnation_count = 0
        best_ever = float("-inf")
        
        for gen in range(generations):
            best_team, best_fitness = self.evolve_generation()
            
            # Check for improvement
            if best_fitness > best_ever:
                best_ever = best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Log progress
            if verbose and (gen + 1) % 10 == 0:
                avg_fitness = np.mean([
                    self.fitness_fn(t) for t in self.population[:20]
                ])
                logger.info(
                    f"Generation {gen + 1}: "
                    f"Best={best_fitness:.1f}, "
                    f"Avg={avg_fitness:.1f}, "
                    f"Stagnation={stagnation_count}"
                )
            
            # Early stopping on stagnation
            if stagnation_count >= self.config.stagnation_limit:
                logger.info(f"Stopping early due to stagnation at generation {gen + 1}")
                break
        
        # Save results
        self._save_results()
        
        logger.info(f"Evolution complete! Best fitness: {best_ever:.1f}")
        return self.best_team
    
    def _save_results(self):
        """Save evolution results to disk."""
        results = {
            "generation": self.generation,
            "best_team": self.best_team.to_dict() if self.best_team else None,
            "fitness_history": self.best_fitness_history,
            "config": {
                "population_size": self.config.population_size,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate,
            }
        }
        
        results_file = self.save_dir / "evolution_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save best team as Showdown paste
        if self.best_team:
            paste_file = self.save_dir / "best_team.txt"
            with open(paste_file, "w") as f:
                f.write(self.best_team.to_showdown_paste())
        
        logger.info(f"Results saved to {self.save_dir}")


def main():
    """Run team evolution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evolve VGC teams")
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Population size"
    )
    
    args = parser.parse_args()
    
    config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population_size,
    )
    
    evolver = TeamEvolver(config=config)
    best_team = evolver.run()
    
    logger.info("\nBest Team Found:")
    for i, p in enumerate(best_team.pokemon):
        logger.info(f"  {i+1}. {p.species} @ {p.item}")


if __name__ == "__main__":
    main()

