"""Column layouts of the packed agent / fauna state arrays (compile-time constants for Numba)."""

# agent float32 state  A[N, NA]
A_ENERGY, A_HYDRATION, A_HEALTH, A_WARMTH, A_CARRY_AMT, A_TEMP_EFF = range(6)
NA = 6

# agent int64 state  I[N, NI]
(I_ALIVE, I_X, I_Y, I_AGE, I_LIFESPAN, I_UID, I_PARENT, I_GEN, I_CARRY_TYPE, I_LAST_ACTION,
 I_SIGNAL, I_REPRO_COOLDOWN, I_LAST_CAUSE, I_BORN, I_SHELTERED, I_HIST_PTR, I_DIED_NOW,
 I_BORN_NOW, I_REPRODUCED_NOW) = range(19)
NI = 19

# agent float64 lifetime statistics  S[N, NS]
(S_ENERGY_EATEN, S_WATER_DRUNK, S_DISTANCE, S_KILL_SMALL, S_KILL_LARGE, S_FOOD_DROPPED,
 S_FOOD_FROM_OTHERS, S_STRIKES, S_DAMAGE_TAKEN, S_SIGNALS, S_BUILDS, S_CHILDREN) = range(12)
NS = 12
STAT_NAMES = ["energy_eaten", "water_drunk", "distance", "kill_small", "kill_large", "food_dropped",
              "food_from_others", "strikes", "damage_taken", "signals", "builds", "children"]

# fauna float32  FF[M, NFF]
F_ENERGY, F_HEALTH = range(2)
NFF = 2
# fauna int64  FI[M, NFI]; attacker slots hold agent uids and ticks for the cooperative-kill rule
(F_ALIVE, F_X, F_Y, F_SPECIES, F_AGE, F_FLEE, F_ATT0, F_ATT1, F_ATT2, F_ATT3,
 F_ATT_T0, F_ATT_T1, F_ATT_T2, F_ATT_T3) = range(14)
NFI = 14
N_ATT_SLOTS = 4

# event record  EV[k] = (tick, type, a_uid, b_uid, x, y, value)
EV_COLS = 7

# packed scalar parameter vector P (float64) -- built by env from config each tick
(P_SEED, P_TICK, P_T_GLOBAL, P_LAPSE, P_SEA, P_RAINING, P_NIGHT, P_COMFORT, P_COLD, P_HEAT,
 P_BASAL, P_MOVE, P_CARRY, P_REST, P_HYD, P_HEAT_MULT, P_WARMTH_RATE, P_HUNGER_DMG, P_COLD_DMG,
 P_FIRE_DMG, P_HEAL, P_DRINK, P_CARRY_CAP, P_STRIKE_DMG, P_PLANT_BITE, P_PLANT_E, P_FRUIT_E,
 P_MEAT_BITE, P_MEAT_E, P_RAIN_DRINK, P_N_SYMBOLS, P_SIGNAL_COST, P_NICHE, P_AGGRESSION,
 P_COMMS, P_HUDDLE, P_COOP_LARGE, P_MIN_ATTACKERS, P_ATT_WINDOW, P_FRUIT_MAX, P_HIST_LEN,
 P_DROUGHT, P_SMALL_E, P_LARGE_E, P_PRED_DMG, P_FLEE_R, P_PRED_SENSE, P_FAUNA_MAX0, P_FAUNA_MAX1,
 P_FAUNA_MAX2, P_FAUNA_MIN0, P_FAUNA_MIN1, P_FAUNA_MIN2, P_BUILD_IDX, P_STRIKE_IDX, P_SIGNAL_IDX,
 P_SNOWING, P_PRED_ATTACK, P_FAUNA_COLD, P_FAUNA_BREED) = range(60)
NP = 60

# fauna species parameters  FP[species, NFP] (float64, from config fauna.<species>.*)
(FP_METAB, FP_BITE, FP_CONV, FP_SAT, FP_REPRO_THR, FP_REPRO_RATE, FP_MAX_AGE, FP_GAIN_SMALL,
 FP_GAIN_LARGE, FP_GAIN_AGENT, FP_BREED_FORAGE, FP_BROWSE) = range(12)
NFP = 12

# fauna demographic counters  FSTAT[species, NFS] (cumulative int64)
FS_BIRTH, FS_IMMIGRANT, FS_STARVED, FS_OLD, FS_FIRE, FS_EATEN, FS_HUNTED = range(7)
NFS = 7
FSTAT_NAMES = ["births", "immigrants", "starved", "old_age", "fire", "eaten", "hunted"]
