"""SQLAlchemy database models for storing Pokemon battle data."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Boolean,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from src.config import config

Base = declarative_base()


class Pokemon(Base):
    """Pokemon species data."""
    
    __tablename__ = "pokemon"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    dex_number = Column(Integer)
    type1 = Column(String(20))
    type2 = Column(String(20), nullable=True)
    
    # Base stats
    hp = Column(Integer)
    attack = Column(Integer)
    defense = Column(Integer)
    special_attack = Column(Integer)
    special_defense = Column(Integer)
    speed = Column(Integer)
    
    # Available abilities (JSON list)
    abilities = Column(JSON)
    
    # Metadata
    generation = Column(Integer)
    is_legendary = Column(Boolean, default=False)
    is_mythical = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Pokemon(name='{self.name}')>"


class Move(Base):
    """Move data."""
    
    __tablename__ = "moves"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    type = Column(String(20))
    category = Column(String(20))  # Physical, Special, Status
    power = Column(Integer, nullable=True)
    accuracy = Column(Integer, nullable=True)
    pp = Column(Integer)
    priority = Column(Integer, default=0)
    target = Column(String(50))  # single, all-adjacent, self, etc.
    effect = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<Move(name='{self.name}')>"


class Item(Base):
    """Held item data."""
    
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    effect = Column(Text)
    
    def __repr__(self):
        return f"<Item(name='{self.name}')>"


class Ability(Base):
    """Ability data."""
    
    __tablename__ = "abilities"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    effect = Column(Text)
    
    def __repr__(self):
        return f"<Ability(name='{self.name}')>"


class Team(Base):
    """Stored team composition."""
    
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    format = Column(String(50))  # e.g., "gen9vgc2024regg"
    
    # Team data stored as JSON
    pokemon_data = Column(JSON)  # List of 6 Pokemon with full sets
    
    # Performance metrics
    elo_rating = Column(Float, default=1500.0)
    games_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    
    # Source
    source = Column(String(50))  # "generated", "scraped", "user"
    source_url = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def __repr__(self):
        return f"<Team(name='{self.name}', elo={self.elo_rating:.0f})>"


class BattleReplay(Base):
    """Stored battle replay data."""
    
    __tablename__ = "battle_replays"
    
    id = Column(Integer, primary_key=True)
    replay_id = Column(String(100), unique=True, nullable=False)
    format = Column(String(50))
    
    # Players
    player1_name = Column(String(100))
    player2_name = Column(String(100))
    winner = Column(String(100))
    
    # Team data
    player1_team = Column(JSON)
    player2_team = Column(JSON)
    
    # Battle log
    battle_log = Column(Text)
    
    # Parsed turns data
    turns_data = Column(JSON)
    total_turns = Column(Integer)
    
    # Metadata
    upload_time = Column(DateTime)
    rating = Column(Integer, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<BattleReplay(id='{self.replay_id}')>"


class UsageStats(Base):
    """Pokemon usage statistics from Smogon."""
    
    __tablename__ = "usage_stats"
    
    id = Column(Integer, primary_key=True)
    pokemon_name = Column(String(100), nullable=False)
    format = Column(String(50), nullable=False)
    month = Column(String(10))  # e.g., "2024-06"
    
    # Usage data
    usage_percent = Column(Float)
    raw_count = Column(Integer)
    
    # Common sets
    common_items = Column(JSON)  # {item: percentage}
    common_abilities = Column(JSON)
    common_moves = Column(JSON)
    common_teammates = Column(JSON)
    common_spreads = Column(JSON)  # EV spreads
    common_tera_types = Column(JSON)
    
    # Checks and counters
    checks_counters = Column(JSON)
    
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UsageStats(pokemon='{self.pokemon_name}', usage={self.usage_percent:.1f}%)>"


class ModelCheckpoint(Base):
    """Trained model checkpoints."""
    
    __tablename__ = "model_checkpoints"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    model_type = Column(String(50))  # "battle_ai", "team_builder"
    
    # Path to saved model
    file_path = Column(String(500))
    
    # Training info
    training_steps = Column(Integer)
    elo_rating = Column(Float)
    
    # Performance metrics
    win_rate_vs_random = Column(Float)
    win_rate_vs_heuristic = Column(Float)
    
    # Config used
    training_config = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelCheckpoint(name='{self.name}', elo={self.elo_rating:.0f})>"


def get_engine():
    """Get SQLAlchemy engine."""
    return create_engine(config.database.connection_string)


def get_session():
    """Get database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_database():
    """Initialize database with all tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"Database initialized at {config.database.db_path}")


if __name__ == "__main__":
    init_database()

