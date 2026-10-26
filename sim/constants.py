"""Enumerations shared by the engine, analysis and viewer. Values are part of the log format."""

# --- terrain classes (uint8) -------------------------------------------------
DEEP_WATER = 0   # ocean / deep lake: impassable, drinkable from adjacent cells
SHALLOW = 1      # river / shore / shallow lake: passable (slow), drinkable
SAND = 2
GRASS = 3        # open plains
MEADOW = 4       # fertile
FOREST = 5       # wood, fruit trees, partial occlusion
BARREN = 6       # dry, low fertility
ROCK = 7         # hills: passable (slow), stone
MOUNTAIN = 8     # impassable, opaque
CAVE = 9         # shelter: warmth + rain cover
N_TERRAIN = 10
TERRAIN_NAMES = ["deep_water", "shallow", "sand", "grass", "meadow", "forest",
                 "barren", "rock", "mountain", "cave"]

# --- actions ------------------------------------------------------------------
NOOP, MOVE_N, MOVE_E, MOVE_S, MOVE_W, REST, EAT, DRINK, HUNT, PICKUP, DROP = range(11)
N_BASE_ACTIONS = 11
BASE_ACTION_NAMES = ["noop", "move_n", "move_e", "move_s", "move_w", "rest", "eat",
                     "drink", "hunt", "pickup", "drop"]
# optional actions are appended in this order when enabled: build, strike, signal_0..signal_{S-1}

# --- carried item types --------------------------------------------------------
CARRY_NONE, CARRY_FOOD, CARRY_WOOD, CARRY_STONE = range(4)

# --- fauna species --------------------------------------------------------------
SMALL_PREY, LARGE_PREY, PREDATOR = range(3)

# --- weather ---------------------------------------------------------------------
CLEAR, RAIN, FOG = range(3)

# --- observation grid channels ---------------------------------------------------
OBS_CHANNELS = ["visible", "blocked", "water", "forest", "rock", "shelter", "plant",
                "fruit", "food_item", "small_prey", "large_prey", "predator",
                "agent", "fire", "signal", "snow"]
N_OBS_CHANNELS = len(OBS_CHANNELS)
SELF_FEATURES = ["energy", "hydration", "health", "warmth", "age_frac", "carry_food",
                 "carry_wood", "carry_stone", "temperature", "light", "raining",
                 "sheltered", "maturity"]
NEIGHBOR_FEATURES = ["present", "dx", "dy", "health", "carry_food", "last_action",
                     "share_rate", "strike_rate", "kin"]

# --- events (log format) -----------------------------------------------------------
EV_BIRTH, EV_DEATH, EV_EAT_PLANT, EV_EAT_FRUIT, EV_EAT_MEAT, EV_KILL_SMALL, EV_KILL_LARGE, \
    EV_PICKUP_FOOD, EV_DROP_FOOD, EV_SIGNAL, EV_BUILD, EV_STRIKE, EV_PREDATOR_ATTACK, \
    EV_FIRE_START, EV_FLOOD_START, EV_FAMINE_START, EV_DROUGHT_START, EV_COLD_SNAP_START, \
    EV_DRINK, EV_ATTACK_LARGE = range(20)
EVENT_NAMES = ["birth", "death", "eat_plant", "eat_fruit", "eat_meat", "kill_small", "kill_large",
               "pickup_food", "drop_food", "signal", "build", "strike", "predator_attack",
               "fire_start", "flood_start", "famine_start", "drought_start", "cold_snap_start",
               "drink", "attack_large"]

# --- causes of death -------------------------------------------------------------------
DEATH_STARVATION, DEATH_THIRST, DEATH_COLD, DEATH_FIRE, DEATH_PREDATOR, DEATH_OLD_AGE, \
    DEATH_STRIKE, DEATH_OTHER = range(8)
DEATH_NAMES = ["starvation", "thirst", "cold", "fire", "predator", "old_age", "strike", "other"]
