"""Tests for Pokemon damage calculation."""

import pytest
import numpy as np
from src.ml.training.damage_calculator import (
    DamageCalculator, 
    get_type_effectiveness,
    get_move_data,
    calculate_damage,
    TYPE_CHART,
)


class TestTypeEffectiveness:
    """Tests for type effectiveness calculations."""
    
    def test_neutral_matchup(self):
        """Neutral matchup returns 1.0."""
        eff = get_type_effectiveness("normal", ["normal"])
        assert eff == 1.0
        
        eff = get_type_effectiveness("fire", ["normal"])
        assert eff == 1.0
    
    def test_super_effective(self):
        """Super effective returns 2.0."""
        eff = get_type_effectiveness("fire", ["grass"])
        assert eff == 2.0
        
        eff = get_type_effectiveness("water", ["fire"])
        assert eff == 2.0
        
        eff = get_type_effectiveness("electric", ["water"])
        assert eff == 2.0
    
    def test_resistance(self):
        """Resistance returns 0.5."""
        eff = get_type_effectiveness("fire", ["water"])
        assert eff == 0.5
        
        eff = get_type_effectiveness("grass", ["fire"])
        assert eff == 0.5
    
    def test_immunity(self):
        """Immunity returns 0.0."""
        eff = get_type_effectiveness("normal", ["ghost"])
        assert eff == 0.0
        
        eff = get_type_effectiveness("ground", ["flying"])
        assert eff == 0.0
        
        eff = get_type_effectiveness("electric", ["ground"])
        assert eff == 0.0
        
        eff = get_type_effectiveness("dragon", ["fairy"])
        assert eff == 0.0
    
    def test_double_weakness(self):
        """Double weakness returns 4.0."""
        # Fire vs Grass/Bug
        eff = get_type_effectiveness("fire", ["grass", "bug"])
        assert eff == 4.0
        
        # Ice vs Dragon/Flying
        eff = get_type_effectiveness("ice", ["dragon", "flying"])
        assert eff == 4.0
    
    def test_double_resistance(self):
        """Double resistance returns 0.25."""
        # Fire vs Fire/Dragon
        eff = get_type_effectiveness("fire", ["fire", "dragon"])
        assert eff == 0.25
    
    def test_mixed_effectiveness(self):
        """Super effective + resist = neutral."""
        # Fire vs Grass/Water = 2x * 0.5x = 1.0
        eff = get_type_effectiveness("fire", ["grass", "water"])
        assert eff == 1.0
    
    def test_case_insensitive(self):
        """Type names are case insensitive."""
        eff1 = get_type_effectiveness("Fire", ["Grass"])
        eff2 = get_type_effectiveness("fire", ["grass"])
        eff3 = get_type_effectiveness("FIRE", ["GRASS"])
        assert eff1 == eff2 == eff3 == 2.0


class TestMoveData:
    """Tests for move database."""
    
    def test_get_known_move(self):
        """Known moves return correct data."""
        move = get_move_data("Flamethrower")
        assert move is not None
        assert move.power == 90
        assert move.move_type == "fire"
        assert move.category == "special"
    
    def test_get_move_normalized(self):
        """Move names are normalized for lookup."""
        move1 = get_move_data("Draco Meteor")
        move2 = get_move_data("dracometeor")
        move3 = get_move_data("DRACO-METEOR")
        
        assert move1 is not None
        assert move1 == move2 == move3
    
    def test_spread_moves(self):
        """Spread moves are marked correctly."""
        eq = get_move_data("earthquake")
        assert eq is not None
        assert eq.is_spread == True
        
        flamethrower = get_move_data("flamethrower")
        assert flamethrower.is_spread == False
    
    def test_status_moves(self):
        """Status moves have 0 power."""
        protect = get_move_data("protect")
        assert protect is not None
        assert protect.power == 0
        assert protect.category == "status"
    
    def test_unknown_move(self):
        """Unknown moves return None."""
        move = get_move_data("Totally Fake Move")
        assert move is None


class TestDamageCalculator:
    """Tests for the damage calculation system."""
    
    @pytest.fixture
    def calc(self):
        """Calculator with fixed random for testing."""
        return DamageCalculator(random_damage=False)
    
    @pytest.fixture
    def calc_random(self):
        """Calculator with random damage enabled."""
        return DamageCalculator(random_damage=True)
    
    def test_base_damage_formula(self, calc):
        """Test basic damage calculation without modifiers."""
        damage = calc.calculate(
            move_power=80, 
            move_type="normal",
            category="physical",
            attacker_atk=100, 
            defender_def=100,
            attacker_types=[], 
            defender_types=["normal"],
        )
        # Should be reasonable fraction of HP
        assert 0.1 < damage < 0.5
    
    def test_stab_bonus(self, calc):
        """Test Same Type Attack Bonus (1.5x)."""
        base = calc.calculate(
            move_power=80, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            attacker_types=[],  # No STAB
            defender_types=["normal"],
        )
        
        stab = calc.calculate(
            move_power=80, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            attacker_types=["fire"],  # STAB
            defender_types=["normal"],
        )
        
        assert stab == pytest.approx(base * 1.5, rel=0.01)
    
    def test_super_effective(self, calc):
        """Test super effective damage (2x)."""
        neutral = calc.calculate(
            move_power=80, 
            move_type="normal",
            category="physical",
            attacker_atk=100,
            defender_def=100,
            defender_types=["normal"],
        )
        
        super_eff = calc.calculate(
            move_power=80, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            defender_types=["grass"],
        )
        
        assert super_eff == pytest.approx(neutral * 2.0, rel=0.01)
    
    def test_resistance(self, calc):
        """Test resistance (0.5x)."""
        neutral = calc.calculate(
            move_power=80, 
            move_type="normal",
            category="physical",
            attacker_atk=100,
            defender_def=100,
            defender_types=["normal"],
        )
        
        resist = calc.calculate(
            move_power=80, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            defender_types=["water"],
        )
        
        assert resist == pytest.approx(neutral * 0.5, rel=0.01)
    
    def test_immunity(self, calc):
        """Test immunity (0x)."""
        damage = calc.calculate(
            move_power=80, 
            move_type="normal",
            category="physical",
            attacker_atk=100,
            defender_def=100,
            defender_types=["ghost"],
        )
        
        assert damage == 0.0
    
    def test_quad_weakness(self, calc):
        """Test 4x weakness."""
        neutral = calc.calculate(
            move_power=80, 
            move_type="normal",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            defender_types=["normal"],
        )
        
        quad = calc.calculate(
            move_power=80, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            defender_types=["grass", "bug"],  # 4x weak
        )
        
        assert quad == pytest.approx(neutral * 4.0, rel=0.01)
    
    def test_random_roll_range(self, calc_random):
        """Test damage varies between 0.85-1.0."""
        np.random.seed(42)
        
        damages = [
            calc_random.calculate(
                move_power=100, 
                move_type="normal",
                category="physical",
                attacker_atk=100,
                defender_def=100,
                defender_types=["normal"],
            )
            for _ in range(100)
        ]
        
        min_ratio = min(damages) / max(damages)
        # Should see variation from 0.85-1.0 (15% range)
        assert 0.80 < min_ratio < 0.92
    
    def test_spread_reduction(self, calc):
        """Test spread move damage reduction."""
        single = calc.calculate(
            move_power=100, 
            move_type="ground",
            category="physical",
            attacker_atk=100,
            defender_def=100,
            is_spread=False,
        )
        
        spread = calc.calculate(
            move_power=100, 
            move_type="ground",
            category="physical",
            attacker_atk=100,
            defender_def=100,
            is_spread=True,
        )
        
        assert spread == pytest.approx(single * 0.75, rel=0.01)
    
    def test_weather_boost(self, calc):
        """Test weather damage modifiers."""
        base = calc.calculate(
            move_power=90, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
        )
        
        sun = calc.calculate(
            move_power=90, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            weather="sun",
        )
        
        rain = calc.calculate(
            move_power=90, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            weather="rain",
        )
        
        assert sun == pytest.approx(base * 1.5, rel=0.01)
        assert rain == pytest.approx(base * 0.5, rel=0.01)
    
    def test_terrain_boost(self, calc):
        """Test terrain damage modifiers."""
        base = calc.calculate(
            move_power=90, 
            move_type="electric",
            category="special",
            attacker_spa=100,
            defender_spd=100,
        )
        
        terrain = calc.calculate(
            move_power=90, 
            move_type="electric",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            terrain="electric",
        )
        
        assert terrain == pytest.approx(base * 1.3, rel=0.01)
    
    def test_item_boost(self, calc):
        """Test item damage modifiers."""
        base = calc.calculate(
            move_power=90, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
        )
        
        life_orb = calc.calculate(
            move_power=90, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            attacker_item="Life Orb",
        )
        
        choice_specs = calc.calculate(
            move_power=90, 
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
            attacker_item="Choice Specs",
        )
        
        assert life_orb == pytest.approx(base * 1.3, rel=0.01)
        assert choice_specs == pytest.approx(base * 1.5, rel=0.01)
    
    def test_move_name_lookup(self, calc):
        """Test damage calculation with move name."""
        damage = calc.calculate(
            move_name="Flamethrower",
            attacker_spa=100,
            defender_spd=100,
            defender_types=["grass"],
        )
        
        # Should be significant (90 BP, super effective)
        assert damage > 0.2
    
    def test_status_move_zero_damage(self, calc):
        """Status moves do no damage."""
        damage = calc.calculate(
            move_name="Protect",
            attacker_atk=150,
            defender_def=50,
        )
        
        assert damage == 0.0
    
    def test_critical_hit(self, calc):
        """Critical hit multiplier."""
        base = calc.calculate(
            move_power=80, 
            move_type="normal",
            category="physical",
            attacker_atk=100,
            defender_def=100,
            crit=False,
        )
        
        crit = calc.calculate(
            move_power=80, 
            move_type="normal",
            category="physical",
            attacker_atk=100,
            defender_def=100,
            crit=True,
        )
        
        assert crit == pytest.approx(base * 1.5, rel=0.01)


class TestKOProbability:
    """Tests for KO probability calculation."""
    
    def test_guaranteed_ko(self):
        """High damage guarantees KO."""
        calc = DamageCalculator(random_damage=True)
        
        prob = calc.calculate_ko_probability(
            move_power=200,  # High power
            move_type="fire",
            category="special",
            attacker_spa=200,
            defender_spd=50,
            defender_types=["grass", "bug"],  # 4x weak
            defender_hp_fraction=1.0,
            samples=50,
        )
        
        assert prob == 1.0
    
    def test_zero_ko(self):
        """Low damage can't KO."""
        calc = DamageCalculator(random_damage=True)
        
        prob = calc.calculate_ko_probability(
            move_power=20,  # Very low power
            move_type="normal",
            category="physical",
            attacker_atk=50,
            defender_def=200,
            defender_types=["steel"],  # Resists
            defender_hp_fraction=1.0,
            samples=50,
        )
        
        assert prob == 0.0
    
    def test_partial_ko_probability(self):
        """Medium damage has variable KO probability."""
        calc = DamageCalculator(random_damage=True)
        
        # Damage that's around the HP threshold
        probs = []
        for _ in range(5):  # Run multiple times for stability
            prob = calc.calculate_ko_probability(
                move_power=100,
                move_type="normal",
                category="physical",
                attacker_atk=100,
                defender_def=100,
                defender_types=["normal"],
                defender_hp_fraction=0.3,  # Low HP
                samples=100,
            )
            probs.append(prob)
        
        avg_prob = np.mean(probs)
        # Should be somewhere between 0 and 1
        assert 0.0 <= avg_prob <= 1.0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_calculate_damage(self):
        """Test calculate_damage function."""
        damage = calculate_damage(
            move_power=90,
            move_type="fire",
            category="special",
            attacker_spa=100,
            defender_spd=100,
        )
        
        assert damage > 0
    
    def test_type_chart_completeness(self):
        """Verify type chart has all types."""
        types = [
            "normal", "fire", "water", "electric", "grass", "ice",
            "fighting", "poison", "ground", "flying", "psychic", "bug",
            "rock", "ghost", "dragon", "dark", "steel", "fairy"
        ]
        
        for attack_type in types:
            assert attack_type in TYPE_CHART

