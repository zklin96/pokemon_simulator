"""Main Streamlit application for Pokemon VGC AI."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config, MODELS_DIR, DATA_DIR
from src.ml.team_builder.team import Team, PokemonSet
from src.ml.team_builder.vgc_data import VGC_POKEMON_POOL, VGC_ITEMS, TERA_TYPES
from src.app.components.battle_viz import (
    render_battle_field, render_battle_log, render_action_selector,
    create_demo_battle_state, BattleState, PokemonState
)

# Page configuration
st.set_page_config(
    page_title="Pokemon VGC AI",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e94560, #f39c12, #00d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .pokemon-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    
    .win-indicator {
        color: #00ff88;
        font-weight: bold;
    }
    
    .loss-indicator {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">⚔️ Pokemon VGC AI ⚔️</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #aaa; font-size: 1.2rem;'>"
        "Battle Simulator & Team Optimizer powered by Reinforcement Learning"
        "</p>",
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🎮 Battle Arena", "📊 Team Builder", "📈 Analytics", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AI ELO", "1650", "+50")
        with col2:
            st.metric("Win Rate", "62%", "+5%")
        
        st.divider()
        st.markdown("##### Built with ❤️ using")
        st.markdown("- poke-env\n- Stable-Baselines3\n- Streamlit")
        
    return page


def render_home():
    """Render the home page."""
    st.header("Welcome to Pokemon VGC AI!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <h2>🎮</h2>
            <h3>Battle Arena</h3>
            <p>Fight against our AI with your custom team</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <h2>📊</h2>
            <h3>Team Builder</h3>
            <p>Build and optimize teams using AI suggestions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <h2>📈</h2>
            <h3>Analytics</h3>
            <p>View metagame stats and matchup analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Recent Activity
    st.subheader("📝 Recent Activity")
    
    activity_data = pd.DataFrame({
        "Time": ["5 min ago", "15 min ago", "1 hour ago", "2 hours ago"],
        "Event": ["Battle Won", "Team Created", "Model Updated", "Battle Lost"],
        "Details": [
            "vs RandomAgent (1650 ELO)",
            "Flutter Mane / Rillaboom core",
            "Self-play iteration #42",
            "vs HeuristicAgent (1580 ELO)"
        ]
    })
    st.dataframe(activity_data, hide_index=True, use_container_width=True)


def render_battle_arena():
    """Render the battle arena page."""
    st.header("🎮 Battle Arena")
    
    # Toggle between setup and battle views
    view_mode = st.radio(
        "Mode",
        ["Setup", "Battle Demo"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if view_mode == "Setup":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Team")
            
            # Team selector
            team_option = st.radio(
                "Select Team",
                ["Use Random Team", "Build Custom Team", "Load Saved Team"]
            )
            
            if team_option == "Use Random Team":
                if st.button("🎲 Generate Random Team"):
                    team = Team.random()
                    st.session_state["user_team"] = team
            
            # Display current team
            if "user_team" in st.session_state:
                team = st.session_state["user_team"]
                for i, pokemon in enumerate(team.pokemon):
                    with st.expander(f"**{i+1}. {pokemon.species}**", expanded=i < 2):
                        st.write(f"**Item:** {pokemon.item}")
                        st.write(f"**Tera Type:** {pokemon.tera_type}")
                        st.write(f"**Nature:** {pokemon.nature}")
                        st.write(f"**Moves:** {', '.join(pokemon.moves)}")
        
        with col2:
            st.subheader("AI Opponent")
            
            # AI selector
            ai_level = st.selectbox(
                "AI Difficulty",
                ["Random (Easy)", "Heuristic (Medium)", "Trained RL Agent (Hard)"]
            )
            
            st.info(f"Selected: **{ai_level}**")
            
            # Battle status
            st.subheader("Battle Options")
            
            battle_format = st.selectbox(
                "Format",
                ["VGC 2024 Reg G", "VGC 2024 Reg H", "VGC 2025 Reg G"]
            )
            
            if st.button("⚔️ Start Battle!", type="primary", use_container_width=True):
                st.balloons()
                st.session_state["in_battle"] = True
                st.success("Battle simulation started!")
                st.info("Switch to 'Battle Demo' to see the visualization.")
    
    else:  # Battle Demo
        st.info("🎮 Demo Battle Visualization (not connected to live game)")
        
        # Create demo battle state
        demo_state = create_demo_battle_state()
        
        # Render battle field
        render_battle_field(demo_state)
        
        # Render battle log
        render_battle_log(demo_state.log)
        
        # Action selector
        st.divider()
        moves_demo = ["Shadow Ball", "Moonblast", "Protect", "Icy Wind",
                      "Grassy Glide", "Wood Hammer", "Fake Out", "U-turn"]
        bench_demo = ["Incineroar", "Amoonguss"]
        actions = render_action_selector(moves_demo, bench_demo, can_tera=True)


def render_team_builder():
    """Render the team builder page."""
    st.header("📊 Team Builder")
    
    tab1, tab2, tab3 = st.tabs(["Build Team", "AI Suggestions", "Saved Teams"])
    
    with tab1:
        st.subheader("Build Your Team")
        
        # Initialize session state for team
        if "building_team" not in st.session_state:
            st.session_state["building_team"] = []
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pokemon selector
            for i in range(6):
                with st.expander(f"Pokemon Slot {i+1}", expanded=i == 0):
                    selected = st.selectbox(
                        "Species",
                        ["Select..."] + VGC_POKEMON_POOL,
                        key=f"pokemon_{i}"
                    )
                    
                    if selected != "Select...":
                        item = st.selectbox(
                            "Held Item",
                            VGC_ITEMS,
                            key=f"item_{i}"
                        )
                        
                        tera = st.selectbox(
                            "Tera Type",
                            TERA_TYPES,
                            key=f"tera_{i}"
                        )
        
        with col2:
            st.subheader("Team Summary")
            st.info("Select 6 Pokemon to complete your team")
            
            if st.button("💾 Save Team"):
                st.success("Team saved!")
    
    with tab2:
        st.subheader("AI-Powered Team Suggestions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            core = st.multiselect(
                "Select core Pokemon (1-3)",
                VGC_POKEMON_POOL,
                max_selections=3
            )
        
        with col2:
            strategy = st.selectbox(
                "Team Strategy",
                ["Balanced", "Hyper Offense", "Bulky Offense", "Trick Room", "Weather"]
            )
        
        if st.button("🤖 Generate AI Suggestions"):
            st.info("Running evolutionary team optimization...")
            
            # Mock suggestions
            suggestions = [
                {"pokemon": "Incineroar", "reason": "Provides Intimidate support and Fake Out pressure"},
                {"pokemon": "Rillaboom", "reason": "Grassy Terrain synergy and priority Grassy Glide"},
                {"pokemon": "Amoonguss", "reason": "Rage Powder redirection and Spore utility"},
            ]
            
            for s in suggestions:
                st.success(f"**{s['pokemon']}**: {s['reason']}")
    
    with tab3:
        st.subheader("Saved Teams")
        
        # Mock saved teams
        saved_teams = [
            {"name": "Rain Team", "core": "Pelipper + Palafin", "elo": 1580},
            {"name": "Sun Offense", "core": "Torkoal + Flutter Mane", "elo": 1620},
            {"name": "Trick Room", "core": "Indeedee-F + Ursaluna", "elo": 1550},
        ]
        
        for team in saved_teams:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{team['name']}**")
                    st.caption(f"Core: {team['core']}")
                with col2:
                    st.write(f"ELO: {team['elo']}")
                with col3:
                    st.button("Load", key=f"load_{team['name']}")
                st.divider()


def render_analytics():
    """Render the analytics page."""
    st.header("📈 Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Metagame Stats", "AI Performance", "Matchups"])
    
    with tab1:
        st.subheader("Top Pokemon Usage")
        
        # Mock usage data
        usage_data = pd.DataFrame({
            "Pokemon": ["Flutter Mane", "Incineroar", "Rillaboom", "Urshifu", "Landorus", 
                       "Amoonguss", "Ogerpon", "Tornadus", "Chi-Yu", "Kingambit"],
            "Usage %": [45.2, 42.1, 38.7, 35.4, 32.1, 28.9, 27.5, 25.3, 24.1, 22.8],
            "Win Rate": [52.1, 51.3, 53.2, 49.8, 50.5, 54.2, 55.1, 48.9, 51.7, 53.4]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(usage_data.set_index("Pokemon")["Usage %"])
        
        with col2:
            st.dataframe(usage_data, hide_index=True, use_container_width=True)
    
    with tab2:
        st.subheader("AI Training Progress")
        
        # Mock training data
        training_steps = list(range(0, 100001, 10000))
        elo_history = [1500, 1520, 1545, 1560, 1590, 1605, 1620, 1635, 1650, 1658, 1665]
        
        chart_data = pd.DataFrame({
            "Training Steps": training_steps,
            "ELO": elo_history
        })
        
        st.line_chart(chart_data.set_index("Training Steps"))
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current ELO", "1665", "+15")
        with col2:
            st.metric("Win Rate", "62.4%", "+2.1%")
        with col3:
            st.metric("Games Played", "1,247")
        with col4:
            st.metric("Self-Play Iterations", "42")
    
    with tab3:
        st.subheader("Matchup Analysis")
        
        # Mock matchup matrix
        pokemon_list = ["Flutter Mane", "Incineroar", "Rillaboom", "Urshifu", "Landorus"]
        matchup_matrix = np.random.rand(5, 5) * 40 + 30  # 30-70% win rates
        np.fill_diagonal(matchup_matrix, 50)  # 50% against self
        
        matchup_df = pd.DataFrame(
            matchup_matrix,
            index=pokemon_list,
            columns=pokemon_list
        )
        
        st.dataframe(
            matchup_df.style.background_gradient(cmap="RdYlGn", vmin=30, vmax=70),
            use_container_width=True
        )
        
        st.caption("Values show win rate % when row Pokemon faces column Pokemon")


def render_settings():
    """Render the settings page."""
    st.header("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["General", "Training", "Data"])
    
    with tab1:
        st.subheader("General Settings")
        
        battle_format = st.selectbox(
            "Default Battle Format",
            ["gen9vgc2024regg", "gen9vgc2024regf", "gen9vgc2024rege"]
        )
        
        animation_speed = st.slider("Animation Speed", 0.5, 2.0, 1.0)
        
        st.checkbox("Enable Sound Effects", value=False)
        st.checkbox("Show Damage Calculations", value=True)
    
    with tab2:
        st.subheader("Training Configuration")
        
        st.number_input("Training Steps per Iteration", 1000, 100000, 10000, step=1000)
        st.number_input("Population Size", 5, 50, 10)
        st.slider("Learning Rate", 0.0001, 0.01, 0.0003, format="%.4f")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Training", type="primary"):
                st.info("Training would start here...")
        with col2:
            if st.button("Stop Training"):
                st.warning("Training stopped")
    
    with tab3:
        st.subheader("Data Management")
        
        if st.button("Scrape Smogon Stats"):
            st.info("Scraping Smogon usage statistics...")
        
        if st.button("Scrape Showdown Replays"):
            st.info("Scraping Pokemon Showdown replays...")
        
        st.divider()
        
        st.subheader("Database Info")
        st.info(f"Database Path: {DATA_DIR / 'vgc_ai.db'}")


def main():
    """Main application entry point."""
    render_header()
    page = render_sidebar()
    
    if "🏠 Home" in page:
        render_home()
    elif "🎮 Battle Arena" in page:
        render_battle_arena()
    elif "📊 Team Builder" in page:
        render_team_builder()
    elif "📈 Analytics" in page:
        render_analytics()
    elif "⚙️ Settings" in page:
        render_settings()


if __name__ == "__main__":
    main()

