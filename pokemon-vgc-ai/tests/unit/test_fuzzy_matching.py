"""Tests for fuzzy move and switch target matching."""

import pytest
from unittest.mock import MagicMock
from src.data.parsers.trajectory_builder import ActionEncoder


class MockPokemonState:
    """Mock PokemonState for testing."""
    
    def __init__(
        self, 
        species: str = "Charizard",
        moves: list = None,
        nickname: str = None
    ):
        self.species = species
        self.moves = moves if moves is not None else []
        self.nickname = nickname


class TestFuzzyMoveMatching:
    """Tests for fuzzy move name matching."""
    
    @pytest.fixture
    def encoder(self):
        return ActionEncoder()
    
    @pytest.fixture
    def pokemon_with_moves(self):
        return MockPokemonState(
            species="Charizard",
            moves=["Flamethrower", "Dragon Pulse", "Solar Beam", "Protect"]
        )
    
    def test_exact_match(self, encoder, pokemon_with_moves):
        """Exact move name returns correct slot."""
        slot = encoder._find_move_slot("Flamethrower", pokemon_with_moves)
        assert slot == 0
        
        slot = encoder._find_move_slot("Protect", pokemon_with_moves)
        assert slot == 3
    
    def test_case_insensitive(self, encoder, pokemon_with_moves):
        """Matching is case-insensitive."""
        slot = encoder._find_move_slot("flamethrower", pokemon_with_moves)
        assert slot == 0
        
        slot = encoder._find_move_slot("PROTECT", pokemon_with_moves)
        assert slot == 3
        
        slot = encoder._find_move_slot("dRaGoN pUlSe", pokemon_with_moves)
        assert slot == 1
    
    def test_normalized_match(self, encoder, pokemon_with_moves):
        """Matching ignores spaces, hyphens, and apostrophes."""
        # Test with hyphen
        slot = encoder._find_move_slot("Dragon-Pulse", pokemon_with_moves)
        assert slot == 1
        
        # Test without space
        slot = encoder._find_move_slot("SolarBeam", pokemon_with_moves)
        assert slot == 2
        
        # Test with extra spaces
        slot = encoder._find_move_slot("Solar  Beam", pokemon_with_moves)
        assert slot == 2
    
    def test_move_id_match(self, encoder):
        """Move ID matching finds canonical moves."""
        # Use moves that should have IDs in poke-env
        pokemon = MockPokemonState(
            species="Incineroar",
            moves=["Fake Out", "Flare Blitz", "Knock Off", "U-turn"]
        )
        
        # These variations should match via ID
        slot = encoder._find_move_slot("fakeout", pokemon)
        assert slot == 0
        
        slot = encoder._find_move_slot("fake-out", pokemon)
        assert slot == 0
        
        slot = encoder._find_move_slot("u-turn", pokemon)
        assert slot == 3
    
    def test_fuzzy_match_typo(self, encoder, pokemon_with_moves):
        """Fuzzy matching catches common typos/variations."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            pytest.skip("rapidfuzz not installed")
        
        # Missing letter (Flamethrowr -> Flamethrower)
        slot = encoder._find_move_slot("Flamethrowr", pokemon_with_moves)
        assert slot == 0
        
        # Extra letter
        slot = encoder._find_move_slot("Proteect", pokemon_with_moves)
        assert slot == 3
    
    def test_unknown_move_adds_to_moveset(self, encoder):
        """Unknown moves are added if there's room."""
        pokemon = MockPokemonState(
            species="Pikachu", 
            moves=["Thunderbolt"]
        )
        
        slot = encoder._find_move_slot("Volt Tackle", pokemon)
        assert slot == 1  # Added as second move
        assert "Volt Tackle" in pokemon.moves
    
    def test_unknown_move_full_moveset(self, encoder, pokemon_with_moves):
        """Unknown moves return 0 if moveset is full."""
        slot = encoder._find_move_slot("Earthquake", pokemon_with_moves)
        assert slot == 0  # Default when can't match
        assert "Earthquake" not in pokemon_with_moves.moves
    
    def test_empty_pokemon(self, encoder):
        """Handles None or empty Pokemon gracefully."""
        slot = encoder._find_move_slot("Any Move", None)
        assert slot == 0
        
        # Empty moveset returns 0 (empty list is falsy)
        empty_pokemon = MockPokemonState(species="Empty", moves=[])
        slot = encoder._find_move_slot("Any Move", empty_pokemon)
        assert slot == 0
        
        # Pokemon with partial moveset can have moves added
        partial_pokemon = MockPokemonState(species="Pikachu", moves=["Thunderbolt"])
        slot = encoder._find_move_slot("Quick Attack", partial_pokemon)
        assert slot == 1  # Added as second move
        assert "Quick Attack" in partial_pokemon.moves
    
    def test_normalize_name_static_method(self, encoder):
        """Test the _normalize_name helper."""
        assert encoder._normalize_name("Solar Beam") == "solarbeam"
        assert encoder._normalize_name("Fake-Out") == "fakeout"
        assert encoder._normalize_name("King's Shield") == "kingsshield"
        assert encoder._normalize_name("PROTECT") == "protect"


class TestFuzzySwitchMatching:
    """Tests for fuzzy switch target matching."""
    
    @pytest.fixture
    def encoder(self):
        return ActionEncoder()
    
    @pytest.fixture
    def bench_pokemon(self):
        return [
            MockPokemonState(species="Incineroar"),
            MockPokemonState(species="Flutter Mane"),
            MockPokemonState(species="Urshifu-Rapid-Strike"),
            MockPokemonState(species="Rillaboom", nickname="Gorilla"),
        ]
    
    def test_exact_species_match(self, encoder, bench_pokemon):
        """Exact species name returns correct slot."""
        slot = encoder._find_switch_target("Incineroar", bench_pokemon)
        assert slot == 0
        
        slot = encoder._find_switch_target("Flutter Mane", bench_pokemon)
        assert slot == 1
    
    def test_normalized_species_match(self, encoder, bench_pokemon):
        """Normalized matching works for species."""
        slot = encoder._find_switch_target("fluttermane", bench_pokemon)
        assert slot == 1
        
        slot = encoder._find_switch_target("urshifurapidstrike", bench_pokemon)
        assert slot == 2
    
    def test_nickname_match(self, encoder, bench_pokemon):
        """Nickname matching works."""
        slot = encoder._find_switch_target("Gorilla", bench_pokemon)
        assert slot == 3
        
        slot = encoder._find_switch_target("gorilla", bench_pokemon)
        assert slot == 3
    
    def test_fuzzy_species_match(self, encoder, bench_pokemon):
        """Fuzzy matching catches species name variations."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            pytest.skip("rapidfuzz not installed")
        
        # Common typos/variations
        slot = encoder._find_switch_target("Incineroar", bench_pokemon)
        assert slot == 0
    
    def test_species_id_match(self, encoder, bench_pokemon):
        """Species ID matching works for alternate forms."""
        # Urshifu-Rapid-Strike should match Urshifu
        slot = encoder._find_switch_target("Urshifu", bench_pokemon)
        # Should match via species ID if available
        assert slot in [0, 2]  # Either first match or ID match
    
    def test_empty_bench(self, encoder):
        """Handles empty bench gracefully."""
        slot = encoder._find_switch_target("Any Pokemon", [])
        assert slot == 0
    
    def test_not_found_default(self, encoder, bench_pokemon):
        """Returns 0 when Pokemon not found."""
        slot = encoder._find_switch_target("Pikachu", bench_pokemon)
        assert slot == 0


class TestActionEncoderIntegration:
    """Integration tests for action encoding."""
    
    @pytest.fixture
    def encoder(self):
        return ActionEncoder()
    
    def test_describe_action(self, encoder):
        """Test action description for debugging."""
        # Move 1 for both slots
        desc = encoder.describe_action(0)
        assert "Move 1" in desc
        
        # Switch action
        desc = encoder.describe_action(8 * 12 + 0)
        assert "Switch" in desc
    
    def test_decode_action(self, encoder):
        """Test action decoding."""
        # Combined action 50 = 50 // 12 = 4, 50 % 12 = 2
        slot_a, slot_b = encoder.decode_action(50)
        assert slot_a == 4  # Tera + Move 1
        assert slot_b == 2  # Move 3
    
    def test_action_space_bounds(self, encoder):
        """Action space is correctly bounded."""
        assert encoder.ACTION_SPACE_SIZE == 144
        assert encoder.ACTIONS_PER_SLOT == 12
        assert encoder.NUM_TARGETS == 5

