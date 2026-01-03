"""Battle visualization component for Streamlit dashboard."""

import streamlit as st
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


# Type effectiveness chart (simplified)
TYPE_EFFECTIVENESS = {
    "Fire": {"Grass": 2.0, "Water": 0.5, "Fire": 0.5, "Steel": 2.0, "Ice": 2.0, "Bug": 2.0},
    "Water": {"Fire": 2.0, "Grass": 0.5, "Water": 0.5, "Ground": 2.0, "Rock": 2.0},
    "Grass": {"Water": 2.0, "Fire": 0.5, "Grass": 0.5, "Ground": 2.0, "Rock": 2.0},
    "Electric": {"Water": 2.0, "Flying": 2.0, "Ground": 0.0, "Electric": 0.5},
    "Psychic": {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0},
    "Dark": {"Psychic": 2.0, "Ghost": 2.0, "Fighting": 0.5, "Dark": 0.5, "Fairy": 0.5},
    "Fairy": {"Dragon": 2.0, "Dark": 2.0, "Fighting": 2.0, "Steel": 0.5, "Fire": 0.5, "Poison": 0.5},
    "Ghost": {"Ghost": 2.0, "Psychic": 2.0, "Normal": 0.0, "Dark": 0.5},
    "Dragon": {"Dragon": 2.0, "Fairy": 0.0, "Steel": 0.5},
    "Steel": {"Fairy": 2.0, "Ice": 2.0, "Rock": 2.0, "Steel": 0.5, "Fire": 0.5, "Water": 0.5, "Electric": 0.5},
    "Fighting": {"Normal": 2.0, "Ice": 2.0, "Rock": 2.0, "Dark": 2.0, "Steel": 2.0, "Ghost": 0.0, "Flying": 0.5, "Psychic": 0.5, "Fairy": 0.5},
    "Flying": {"Grass": 2.0, "Fighting": 2.0, "Bug": 2.0, "Rock": 0.5, "Electric": 0.5, "Steel": 0.5},
    "Ground": {"Fire": 2.0, "Electric": 2.0, "Poison": 2.0, "Rock": 2.0, "Steel": 2.0, "Flying": 0.0, "Grass": 0.5, "Bug": 0.5},
    "Rock": {"Fire": 2.0, "Ice": 2.0, "Flying": 2.0, "Bug": 2.0, "Fighting": 0.5, "Ground": 0.5, "Steel": 0.5},
    "Bug": {"Grass": 2.0, "Psychic": 2.0, "Dark": 2.0, "Fire": 0.5, "Fighting": 0.5, "Flying": 0.5, "Ghost": 0.5, "Steel": 0.5, "Fairy": 0.5},
    "Poison": {"Grass": 2.0, "Fairy": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0.0},
    "Ice": {"Grass": 2.0, "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Fire": 0.5, "Water": 0.5, "Ice": 0.5, "Steel": 0.5},
    "Normal": {"Ghost": 0.0, "Rock": 0.5, "Steel": 0.5},
}


@dataclass
class PokemonState:
    """State of a Pokemon in battle."""
    species: str
    current_hp: float  # 0.0 - 1.0
    max_hp: int
    is_active: bool
    is_fainted: bool
    types: List[str]
    status: Optional[str]
    is_tera: bool
    tera_type: Optional[str]
    boosts: Dict[str, int]


@dataclass
class BattleState:
    """Current battle state for visualization."""
    turn: int
    my_team: List[PokemonState]
    opp_team: List[PokemonState]
    weather: Optional[str]
    terrain: Optional[str]
    my_side_conditions: List[str]
    opp_side_conditions: List[str]
    log: List[str]


def render_pokemon_sprite(species: str, is_fainted: bool = False, size: int = 96) -> str:
    """Get Pokemon sprite URL."""
    # Normalize species name for URL
    name_clean = species.lower().replace(" ", "-").replace("'", "")
    # Handle special forms
    form_mapping = {
        "ogerpon-wellspring": "ogerpon-wellspring-mask",
        "ogerpon-hearthflame": "ogerpon-hearthflame-mask",
        "ogerpon-cornerstone": "ogerpon-cornerstone-mask",
        "urshifu": "urshifu-rapid-strike",
    }
    name_clean = form_mapping.get(name_clean, name_clean)
    
    base_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"
    
    # Try official artwork first
    url = f"{base_url}/other/official-artwork/{name_clean.split('-')[0]}.png"
    
    return url


def render_hp_bar(current: float, species: str) -> None:
    """Render an HP bar with color coding."""
    if current <= 0:
        color = "#555"
    elif current < 0.25:
        color = "#ff4444"
    elif current < 0.5:
        color = "#ffaa00"
    else:
        color = "#00cc66"
    
    percent = max(0, min(100, current * 100))
    
    st.markdown(f"""
    <div style="
        background: #333;
        border-radius: 5px;
        height: 12px;
        width: 100%;
        margin: 5px 0;
    ">
        <div style="
            background: {color};
            border-radius: 5px;
            height: 100%;
            width: {percent}%;
            transition: width 0.3s ease;
        "></div>
    </div>
    <p style="text-align: center; margin: 0; color: #aaa; font-size: 0.8rem;">
        {percent:.0f}% HP
    </p>
    """, unsafe_allow_html=True)


def render_pokemon_card(pokemon: PokemonState, is_opponent: bool = False) -> None:
    """Render a Pokemon card with status."""
    opacity = "0.4" if pokemon.is_fainted else "1.0"
    border_color = "#ff4444" if is_opponent else "#00cc66"
    tera_indicator = f"⭐ Tera: {pokemon.tera_type}" if pokemon.is_tera else ""
    
    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid {border_color};
        opacity: {opacity};
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="flex: 1;">
                <h4 style="margin: 0; color: #fff;">
                    {pokemon.species}
                    {"💀" if pokemon.is_fainted else ""}
                    {tera_indicator}
                </h4>
                <p style="margin: 4px 0; color: #888; font-size: 0.9rem;">
                    {" / ".join(pokemon.types)}
                    {f" | {pokemon.status.upper()}" if pokemon.status else ""}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not pokemon.is_fainted:
        render_hp_bar(pokemon.current_hp, pokemon.species)
        
        # Show boosts if any
        boost_str = " ".join([
            f"{'↑' if v > 0 else '↓'}{stat}x{abs(v)}"
            for stat, v in pokemon.boosts.items() if v != 0
        ])
        if boost_str:
            st.caption(boost_str)


def render_field_conditions(weather: Optional[str], terrain: Optional[str], 
                           my_conditions: List[str], opp_conditions: List[str]) -> None:
    """Render field conditions."""
    cols = st.columns(4)
    
    with cols[0]:
        if weather:
            weather_icons = {
                "sun": "☀️", "rain": "🌧️", "sand": "🏜️", "snow": "❄️", "hail": "🌨️"
            }
            icon = weather_icons.get(weather.lower(), "🌤️")
            st.info(f"{icon} {weather.title()}")
    
    with cols[1]:
        if terrain:
            terrain_icons = {
                "grassy": "🌿", "electric": "⚡", "psychic": "🔮", "misty": "💨"
            }
            icon = terrain_icons.get(terrain.lower(), "🏔️")
            st.info(f"{icon} {terrain.title()} Terrain")
    
    with cols[2]:
        if my_conditions:
            st.success(f"🛡️ {', '.join(my_conditions)}")
    
    with cols[3]:
        if opp_conditions:
            st.error(f"⚠️ {', '.join(opp_conditions)}")


def render_battle_field(state: BattleState) -> None:
    """Render the full battle field."""
    st.markdown(f"### Turn {state.turn}")
    
    # Field conditions
    render_field_conditions(
        state.weather, state.terrain,
        state.my_side_conditions, state.opp_side_conditions
    )
    
    st.divider()
    
    # Active Pokemon
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔵 Your Active Pokemon")
        active = [p for p in state.my_team if p.is_active]
        for pokemon in active:
            render_pokemon_card(pokemon, is_opponent=False)
    
    with col2:
        st.markdown("#### 🔴 Opponent Active Pokemon")
        active = [p for p in state.opp_team if p.is_active]
        for pokemon in active:
            render_pokemon_card(pokemon, is_opponent=True)
    
    st.divider()
    
    # Benched Pokemon
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Bench")
        benched = [p for p in state.my_team if not p.is_active and not p.is_fainted]
        for pokemon in benched:
            st.markdown(f"- {pokemon.species} ({pokemon.current_hp*100:.0f}% HP)")
        fainted = [p for p in state.my_team if p.is_fainted]
        if fainted:
            st.caption(f"Fainted: {', '.join(p.species for p in fainted)}")
    
    with col2:
        st.caption("Bench")
        benched = [p for p in state.opp_team if not p.is_active and not p.is_fainted]
        for pokemon in benched:
            st.markdown(f"- {pokemon.species} ({pokemon.current_hp*100:.0f}% HP)")
        fainted = [p for p in state.opp_team if p.is_fainted]
        if fainted:
            st.caption(f"Fainted: {', '.join(p.species for p in fainted)}")


def render_battle_log(log: List[str], max_lines: int = 10) -> None:
    """Render the battle log."""
    with st.expander("📜 Battle Log", expanded=True):
        log_text = "\n".join(log[-max_lines:])
        st.code(log_text, language=None)


def render_action_selector(available_moves: List[str], 
                           can_switch: List[str],
                           can_tera: bool) -> Dict:
    """Render action selection UI."""
    st.markdown("### Choose Your Action")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Slot A (Lead)**")
        action_a = st.radio(
            "Action A",
            ["Move", "Switch", "Tera + Move"],
            key="action_a_type",
            label_visibility="collapsed"
        )
        
        if "Move" in action_a:
            move_a = st.selectbox("Move", available_moves[:4], key="move_a")
            target_a = st.radio("Target", ["Slot 1", "Slot 2", "Both"], key="target_a", horizontal=True)
        elif action_a == "Switch":
            switch_a = st.selectbox("Switch to", can_switch, key="switch_a")
    
    with col2:
        st.markdown("**Slot B**")
        action_b = st.radio(
            "Action B",
            ["Move", "Switch"],
            key="action_b_type",
            label_visibility="collapsed"
        )
        
        if action_b == "Move":
            move_b = st.selectbox("Move", available_moves[4:8] if len(available_moves) > 4 else available_moves[:4], key="move_b")
            target_b = st.radio("Target", ["Slot 1", "Slot 2", "Both"], key="target_b", horizontal=True)
        elif action_b == "Switch":
            switch_b = st.selectbox("Switch to", can_switch, key="switch_b")
    
    if st.button("⚔️ Execute Turn", type="primary", use_container_width=True):
        return {"action_a": action_a, "action_b": action_b}
    
    return {}


def create_demo_battle_state() -> BattleState:
    """Create a demo battle state for testing."""
    return BattleState(
        turn=5,
        my_team=[
            PokemonState(
                species="Flutter Mane",
                current_hp=0.75,
                max_hp=275,
                is_active=True,
                is_fainted=False,
                types=["Ghost", "Fairy"],
                status=None,
                is_tera=False,
                tera_type="Fairy",
                boosts={"spa": 1}
            ),
            PokemonState(
                species="Rillaboom",
                current_hp=0.90,
                max_hp=341,
                is_active=True,
                is_fainted=False,
                types=["Grass"],
                status=None,
                is_tera=False,
                tera_type="Grass",
                boosts={}
            ),
            PokemonState(
                species="Incineroar",
                current_hp=1.0,
                max_hp=362,
                is_active=False,
                is_fainted=False,
                types=["Fire", "Dark"],
                status=None,
                is_tera=False,
                tera_type="Ghost",
                boosts={}
            ),
            PokemonState(
                species="Urshifu",
                current_hp=0.0,
                max_hp=324,
                is_active=False,
                is_fainted=True,
                types=["Fighting", "Dark"],
                status=None,
                is_tera=False,
                tera_type="Dark",
                boosts={}
            ),
        ],
        opp_team=[
            PokemonState(
                species="Chi-Yu",
                current_hp=0.45,
                max_hp=270,
                is_active=True,
                is_fainted=False,
                types=["Dark", "Fire"],
                status=None,
                is_tera=True,
                tera_type="Fire",
                boosts={"spa": 2}
            ),
            PokemonState(
                species="Landorus",
                current_hp=0.60,
                max_hp=319,
                is_active=True,
                is_fainted=False,
                types=["Ground", "Flying"],
                status="paralysis",
                is_tera=False,
                tera_type="Flying",
                boosts={"atk": -1}
            ),
        ],
        weather="Sun",
        terrain="Grassy",
        my_side_conditions=["Tailwind"],
        opp_side_conditions=["Light Screen"],
        log=[
            "Turn 4:",
            "Flutter Mane used Shadow Ball!",
            "Chi-Yu lost 35% HP!",
            "Rillaboom used Grassy Glide!",
            "Landorus lost 40% HP!",
            "Chi-Yu used Heat Wave!",
            "Flutter Mane lost 25% HP!",
            "Landorus used Rock Slide!",
            "Rillaboom lost 10% HP!",
            "Turn 5:",
        ]
    )

