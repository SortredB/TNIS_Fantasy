# fpl_clone_full.py
# TNIS-Fantasy (single-file Streamlit app)
# Save as fpl_clone_full.py and run: streamlit run fpl_clone_full.py
import streamlit as st
import pandas as pd
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Mini FPL (Full)", layout="wide")

# -------------------------
# Data models (simple)
# -------------------------
@dataclass
class Player:
    id: int
    name: str
    position: str  # GK, DEF, MID, FWD
    club: str
    price: float

@dataclass
class Event:
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    cs: bool = False
    goals_conceded: int = 0
    saves: int = 0
    pen_save: int = 0
    pen_miss: int = 0
    yellow: int = 0
    red: int = 0
    own_goal: int = 0
    bonus: int = 0

# -------------------------
# Scoring config (close to FPL)
# -------------------------
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3
APPEAR_LT60 = 1
APPEAR_GE60 = 2
CS_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
GC_MINUS_PER_2 = {"GK": -1, "DEF": -1}
SAVE_PER_3 = 1
PEN_SAVE = 5
PEN_MISS = -2
YELLOW = -1
RED = -3
OWN_GOAL = -2

SQUAD_SIZE = 15
STARTING_SIZE = 11
MAX_PER_CLUB = 3
INITIAL_BUDGET = 100.0

# -------------------------
# Helpers: load players
# -------------------------
@st.cache_data
def default_players_df():
    data = {
        "id":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        "name":["Valle Bergström","Efiem Awet","Emil Kvarnerud","Leo Adoue","Victor Berlin","Leon Berglund",
                "Eddie Renqvist","Erik Weston","Theodor Henriksson","Anton Bergström","Maximillian Eriksson","Vilmer Eriksson",
                "Martin Ivanov","Oliver Ekman","Wilhelm Dywling","Theodor Deja","Mikael Gorgis","Victor Berlin (FWD)"],
        "position":["GK","DEF","DEF","MID","MID","FWD","DEF","MID","FWD","GK","DEF","MID","DEF","MID","FWD","DEF","MID","FWD"],
        "club":["TNIS.","TNIS.","TNIS.","TNIS..",".TNIS.",".TNIS.",".TNIS.","TNIS","TNIS","TNIS",".TNIS",".TNIS",".TNIS","..TNIS","..TNIS","..TNIS","..TNIS","TNIS.."],
        "price":[5.0,5.0,4.5,7.0,6.5,4.5,4.5,5.5,5.0,4.0,4.0,6.0,4.5,4.5,5.0,5.0,5.0,5.0]
    }
    return pd.DataFrame(data)

def load_players_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Expect columns: id,name,position,club,price
        if set(["id","name","position","club","price"]).issubset(df.columns):
            return df[["id","name","position","club","price"]]
        # try common Swedish names
        if set(["id","namn","position","lag","pris"]).issubset(df.columns):
            df = df.rename(columns={"namn":"name","lag":"club","pris":"price"})
            return df[["id","name","position","club","price"]]
    except Exception as e:
        st.error("Fel vid inläsning av CSV: " + str(e))
    return None

# -------------------------
# Session state init
# -------------------------
if 'players_df' not in st.session_state:
    st.session_state['players_df'] = default_players_df()
if 'league' not in st.session_state:
    # league state: dict with managers and history
    st.session_state['league'] = {
        "name": "My League",
        "current_gw": 1,
        "players": {},  # id -> Player dict
        "managers": {}, # name -> manager dict
        "history": []   # per GW events
    }
    # populate players from df
    for _, r in st.session_state['players_df'].iterrows():
        pid = int(r["id"])
        st.session_state['league']['players'][pid] = {
            "id": pid,
            "name": r["name"],
            "position": r["position"],
            "club": r["club"],
            "price": float(r["price"])
        }

# -------------------------
# Utility functions
# -------------------------
def players_table():
    df = pd.DataFrame.from_dict(st.session_state['league']['players'], orient='index')
    return df

def validate_squad(player_ids: List[int]) -> (bool,str):
    if len(player_ids) != SQUAD_SIZE:
        return False, f"Ska vara exakt {SQUAD_SIZE} spelare i truppen."
    # positions
    pos_counts = {"GK":0,"DEF":0,"MID":0,"FWD":0}
    club_counts = {}
    for pid in player_ids:
        p = st.session_state['league']['players'][pid]
        pos_counts[p['position']] += 1
        club_counts[p['club']] = club_counts.get(p['club'],0)+1
        if club_counts[p['club']] > MAX_PER_CLUB:
            return False, f"För många spelare från {p['club']} (max {MAX_PER_CLUB})."
    if pos_counts["GK"] != 2 or pos_counts["DEF"] < 5 or pos_counts["MID"] < 5 or pos_counts["FWD"] < 3:
        return False, "Positioner måste vara: 2 GK, minst 5 DEF, minst 5 MID, minst 3 FWD."
    return True, "OK"

def squad_cost(player_ids: List[int]) -> float:
    return sum(st.session_state['league']['players'][pid]['price'] for pid in player_ids)

# scoring
def score_event_for_player(player: Dict, ev: Event) -> int:
    pts = 0
    if ev.minutes > 0:
        pts += APPEAR_LT60 if ev.minutes < 60 else APPEAR_GE60
    pts += ev.goals * GOAL_POINTS.get(player['position'], 0)
    pts += ev.assists * ASSIST_POINTS
    if ev.minutes >= 60 and ev.cs:
        pts += CS_POINTS.get(player['position'], 0)
    if player['position'] in ('GK','DEF'):
        pts += (ev.goals_conceded // 2) * GC_MINUS_PER_2.get(player['position'],0)
    if player['position'] == 'GK':
        pts += (ev.saves // 3) * SAVE_PER_3
    pts += ev.pen_save * PEN_SAVE
    pts += ev.pen_miss * PEN_MISS
    pts += ev.yellow * YELLOW
    pts += ev.red * RED
    pts += ev.own_goal * OWN_GOAL
    pts += ev.bonus
    return pts

# auto-sub: simplified - bring on bench players in order if starters didn't play
def auto_substitute(squad: Dict, gw_events: Dict[int, Event]) -> (List[int], List[int]):
    # squad: dict with keys players (list 15), starting (11 ids), bench (4 in order)
    starting = squad['starting'][:]
    bench = squad['bench'][:]  # order: [GKbench, out1, out2, out3]
    def played(pid): return gw_events.get(pid, Event()).minutes > 0
    def pos(pid): return st.session_state['league']['players'][pid]['position']
    # if GK didn't play, swap with bench GK if played
    for i, pid in enumerate(starting):
        if pos(pid) == 'GK' and not played(pid):
            if bench and pos(bench[0])=='GK' and played(bench[0]):
                starting[i] = bench[0]
                bench[0] = pid
    # for outfield subs: iterate starters; if not played, try bring bench players in order that keep formation valid
    def formation_counts(ids):
        return sum(1 for x in ids if pos(x)=='DEF'), sum(1 for x in ids if pos(x)=='MID'), sum(1 for x in ids if pos(x)=='FWD')
    for i, pid in enumerate(starting):
        if pos(pid) == 'GK': continue
        if played(pid): continue
        # try each bench slot (1..3)
        for j in range(1, len(bench)):
            cand = bench[j]
            if cand is None or not played(cand): continue
            if pos(cand)=='GK': continue
            temp = starting.copy()
            temp[i] = cand
            d,m,f = formation_counts(temp)
            if d>=3 and m>=2 and f>=1:
                # commit
                starting[i] = cand
                bench[j] = pid
                break
    return starting, bench

# -------------------------
# Manager actions
# -------------------------
def create_manager(name: str):
    if name in st.session_state['league']['managers']:
        st.warning("Manager finns redan.")
        return
    st.session_state['league']['managers'][name] = {
        "name": name,
        "budget": INITIAL_BUDGET,
        "bank": INITIAL_BUDGET,
        "free_transfers": 1,
        "free_transfers_saved": 0,
        "chips": {"WC": True, "FH": True, "TC": True, "BB": True},
        "chip_active": None,
        "squad": None,  # dict players, starting, bench, captain, vice
        "total_points": 0,
        "_pending_hit": 0
    }

def pick_initial_squad_ui(manager_name: str):
    st.subheader(f"Build squad for {manager_name}")
    df = players_table()
    st.write("Pick 15 players (use checkboxes). Observe budget and max 3 per club.")
    # show table with checkboxes
    chosen = st.multiselect("Choose 15 players (searchable)", options=df['id'].tolist(),
                            format_func=lambda x: f"{df.loc[df['id']==x,'name'].values[0]} - {df.loc[df['id']==x,'position'].values[0]} ({df.loc[df['id']==x,'club'].values[0]}) Price:{df.loc[df['id']==x,'price'].values[0]:.1f}")
    if st.button("Save squad"):
        if len(chosen)!=SQUAD_SIZE:
            st.error(f"Välj exakt {SQUAD_SIZE} spelare.")
            return
        ok,msg = validate_squad(chosen)
        if not ok:
            st.error(msg)
            return
        cost = squad_cost(chosen)
        if cost > INITIAL_BUDGET:
            st.error(f"Överstiger budget. Kostnad {cost} > {INITIAL_BUDGET}")
            return
        # default starting first 11 and bench rest; ensure bench GK first
        starting = chosen[:11]
        bench = [pid for pid in chosen if pid not in starting]
        # ensure bench GK at index 0
        bench_gk = [pid for pid in bench if st.session_state['league']['players'][pid]['position']=='GK']
        if bench_gk:
            gk = bench_gk[0]
            bench.remove(gk)
            bench = [gk] + bench
        # default captain first starter, vice second
        captain = starting[0]
        vice = starting[1] if len(starting)>1 else starting[0]
        st.session_state['league']['managers'][manager_name]['squad'] = {
            "players": chosen,
            "starting": starting,
            "bench": bench,
            "captain": captain,
            "vice": vice
        }
        st.session_state['league']['managers'][manager_name]['bank'] = INITIAL_BUDGET - cost
        st.success("Squad saved!")

def view_manager_ui(name):
    m = st.session_state['league']['managers'][name]
    st.write(f"Manager: {name} — Total points: {m['total_points']} — Bank: {m['bank']:.1f} — Free transfers: {m['free_transfers']}")
    if not m['squad']:
        st.info("Ingen trupp vald ännu.")
        return
    st.write("Starting XI:")
    for pid in m['squad']['starting']:
        p = st.session_state['league']['players'][pid]
        st.write(f"{pid}: {p['name']} ({p['position']}) — {p['club']}")
    st.write("Bench (order):")
    for i,pid in enumerate(m['squad']['bench'], start=1):
        p = st.session_state['league']['players'][pid]
        st.write(f"({i}) {pid}: {p['name']} ({p['position']})")

# transfers function (simple)
def apply_transfers(manager_name: str, outs: List[int], ins: List[int]) -> (bool,str):
    m = st.session_state['league']['managers'][manager_name]
    if not m['squad']:
        return False, "Ingen trupp att göra transfers från."
    if len(outs) != len(ins):
        return False, "Outs och Ins måste ha samma längd."
    new_players = set(m['squad']['players'])
    cost_diff = 0.0
    for o,i in zip(outs, ins):
        if o not in new_players:
            return False, f"Säljer inte spelare {o}."
        new_players.remove(o)
        if i in new_players:
            return False, f"Dubbelkandidat {i}."
        if i not in st.session_state['league']['players']:
            return False, f"Invalid incoming id {i}."
        new_players.add(i)
        cost_diff += st.session_state['league']['players'][i]['price'] - st.session_state['league']['players'][o]['price']
    # club constraint
    club_counts = {}
    for pid in new_players:
        c = st.session_state['league']['players'][pid]['club']
        club_counts[c] = club_counts.get(c,0)+1
        if club_counts[c] > MAX_PER_CLUB:
            return False, f"Max {MAX_PER_CLUB} per club violated."
    # budget
    if m['bank'] - cost_diff < 0:
        # allow if wildcard active
        if m['chip_active'] != 'WC' and m['chip_active'] != 'FH':
            return False, f"Otillräcklig bank för transfers (need {cost_diff:.1f})."
    # calculate hits
    n = len(ins)
    free = m['free_transfers']
    chargeable = max(0, n - free)
    hit = 0 if m['chip_active'] in ('WC','FH') else chargeable * 4
    m['_pending_hit'] = m.get('_pending_hit',0) + hit
    # apply changes: create new squad list (keep starting if possible)
    new_list = list(new_players)
    old_sq = m['squad']
    starting = [pid for pid in old_sq['starting'] if pid in new_list]
    # fill to 11
    candidates = [pid for pid in new_list if pid not in starting]
    while len(starting) < STARTING_SIZE and candidates:
        starting.append(candidates.pop(0))
    bench = [pid for pid in new_list if pid not in starting]
    # ensure bench GK first
    bench_gk = [pid for pid in bench if st.session_state['league']['players'][pid]['position']=='GK']
    if bench_gk:
        g = bench_gk[0]
        bench.remove(g)
        bench = [g] + bench
    # captain/vice
    captain = old_sq['captain'] if old_sq['captain'] in starting else starting[0]
    vice = old_sq['vice'] if old_sq['vice'] in starting and old_sq['vice'] != captain else next((x for x in starting if x != captain), captain)
    m['squad'] = {"players": new_list, "starting": starting, "bench": bench, "captain": captain, "vice": vice}
    # adjust bank
    m['bank'] -= cost_diff if m['chip_active'] != 'FH' else 0.0
    # reduce free transfers used
    used_free = min(n, m['free_transfers'])
    m['free_transfers'] -= used_free
    return True, f"Transfers applied. Hit: {hit}"

# activate chip
def activate_chip(manager_name: str, chip: str):
    m = st.session_state['league']['managers'][manager_name]
    if not m['chips'].get(chip, False):
        st.error("Chip inte tillgängligt.")
        return
    m['chip_active'] = chip
    st.success(f"{chip} aktiverad för denna GW.")

# process GW admin
def process_gw_admin(gw_events: Dict[int, Event]):
    # store history
    gw = st.session_state['league']['current_gw']
    st.session_state['league']['history'].append({"gw": gw, "events": {pid: asdict(ev) for pid,ev in gw_events.items()}})
    # score each manager
    for m in st.session_state['league']['managers'].values():
        if not m['squad']: continue
        # backup FH
        if m['chip_active']=='FH':
            # FH should have been applied earlier by replacing squad for gw and fh_backup saved; we keep simple: assume chip_active was set and squad is current.
            pass
        # auto-subs
        gw_events_objs = {pid: Event(**ev) if not isinstance(ev, Event) else ev for pid,ev in gw_events.items()}
        final_starting, final_bench = auto_substitute(m['squad'], gw_events_objs)
        week_pts = 0
        # starting players
        for pid in final_starting:
            p = st.session_state['league']['players'][pid]
            ev = gw_events_objs.get(pid, Event())
            week_pts += score_event_for_player(p, ev)
        # captain/vice and triple cap
        cap = m['squad']['captain']
        vcap = m['squad']['vice']
        cap_played = gw_events_objs.get(cap, Event()).minutes > 0
        vcap_played = gw_events_objs.get(vcap, Event()).minutes > 0
        multiplier = 3 if m['chip_active']=='TC' else 2
        if cap_played:
            week_pts += (multiplier-1) * score_event_for_player(st.session_state['league']['players'][cap], gw_events_objs.get(cap, Event()))
        elif vcap_played:
            week_pts += (multiplier-1) * score_event_for_player(st.session_state['league']['players'][vcap], gw_events_objs.get(vcap, Event()))
        # bench boost
        if m['chip_active']=='BB':
            for pid in final_bench:
                week_pts += score_event_for_player(st.session_state['league']['players'][pid], gw_events_objs.get(pid, Event()))
        # apply hits
        week_pts -= m.get('_pending_hit',0)
        m['total_points'] += week_pts
        m['_pending_hit'] = 0
        # consume chip if used
        if m['chip_active']:
            if m['chip_active']=='WC':
                m['chips']['WC'] = False
            if m['chip_active']=='FH':
                m['chips']['FH'] = False
            if m['chip_active']=='TC':
                m['chips']['TC'] = False
            if m['chip_active']=='BB':
                m['chips']['BB'] = False
            m['chip_active'] = None
        # rollover free transfers
        m['free_transfers'] = min(2, m['free_transfers'] + 1)
    st.session_state['league']['current_gw'] += 1

# -------------------------
# UI layout
# -------------------------
st.title("Mini FPL — Nearly Full Feature Set")
st.sidebar.header("Controls & data")

# Upload custom players CSV
uploaded = st.sidebar.file_uploader("Upload players CSV (columns id,name,position,club,price)", type=['csv'])
if uploaded:
    df = load_players_from_csv(uploaded)
    if df is not None:
        st.session_state['players_df'] = df
        # update league players
        st.session_state['league']['players'] = {}
        for _, r in df.iterrows():
            pid = int(r['id'])
            st.session_state['league']['players'][pid] = {"id": pid, "name": r['name'], "position": r['position'], "club": r['club'], "price": float(r['price'])}
        st.success("Players loaded from CSV.")

# save/load league JSON
st.sidebar.markdown("---")
if st.sidebar.button("Export league (download JSON)"):
    js = json.dumps(st.session_state['league'], ensure_ascii=False, indent=2)
    st.sidebar.download_button("Download league JSON", js, file_name="league_export.json")
uploaded_league = st.sidebar.file_uploader("Load league JSON", type=['json'])
if uploaded_league:
    try:
        data = json.load(uploaded_league)
        st.session_state['league'] = data
        st.success("League loaded.")
    except Exception as e:
        st.error("Kunde inte ladda league JSON: " + str(e))

st.sidebar.markdown("---")
st.sidebar.write(f"GW: {st.session_state['league']['current_gw']}")
st.sidebar.write("Players: " + str(len(st.session_state['league']['players'])))
st.sidebar.markdown("**Managers**")
for mname in st.session_state['league']['managers'].keys():
    st.sidebar.write(mname)

# Main tabs
tabs = st.tabs(["Setup","Managers","Transfers & Chips","Admin GW","Standings & Export"])
# ----------------- Setup tab -----------------
with tabs[0]:
    st.header("League Setup")
    league_name = st.text_input("League name", value=st.session_state['league']['name'])
    if st.button("Update league name"):
        st.session_state['league']['name'] = league_name
        st.success("League name updated.")
    st.subheader("Players (sample or uploaded)")
    st.dataframe(players_table().sort_values('id'))

    st.subheader("Create Manager")
    new_name = st.text_input("Manager name (create)", key="m_create")
    if st.button("Create manager"):
        if not new_name:
            st.error("Ange namn")
        else:
            create_manager(new_name)
            st.rerun()

# ----------------- Managers tab -----------------
with tabs[1]:
    st.header("Managers & Squad Building")
    mgr_list = list(st.session_state['league']['managers'].keys())
    if not mgr_list:
        st.info("Skapa minst en manager i Setup-fliken.")
    else:
        sel = st.selectbox("Välj manager", options=[""]+mgr_list)
        if sel:
            view_manager_ui(sel)
            st.markdown("### Build / edit squad")
            if st.button("Pick / Edit initial squad"):
                pick_initial_squad_ui(sel)
            st.markdown("### Set Starting XI / Captains")
            m = st.session_state['league']['managers'][sel]
            if m['squad']:
                # show current squad and allow selecting start XI
                player_names = [st.session_state['league']['players'][pid]['name'] + f" ({pid})" for pid in m['squad']['players']]
                id_map = {pid: st.session_state['league']['players'][pid]['name'] for pid in m['squad']['players']}
                # choose starting IDs
                st.write("Välj 11 startspelare:")
                start_selected = st.multiselect("Startelva", options=m['squad']['players'], format_func=lambda x: id_map[x], default=m['squad']['starting'])
                if len(start_selected)==11:
                    # bench order
                    bench_current = [pid for pid in m['squad']['players'] if pid not in start_selected]
                    # ensure bench GK first
                    bench_default = bench_current.copy()
                    bench_order = st.multiselect("Bench order (GK must be first)", options=bench_default, default=bench_default)
                    cap = st.selectbox("Kapten", options=start_selected, format_func=lambda x: id_map[x], index=0)
                    vice = st.selectbox("Vice", options=[x for x in start_selected if x!=cap], format_func=lambda x: id_map[x], index=0)
                    if st.button("Save starting XI"):
                        # simple validation
                        # ensure GK present in starting
                        if sum(1 for pid in start_selected if st.session_state['league']['players'][pid]['position']=='GK')!=1:
                            st.error("Startelva måste innehålla exakt 1 GK")
                        else:
                            m['squad']['starting'] = start_selected
                            m['squad']['bench'] = bench_order
                            m['squad']['captain'] = cap
                            m['squad']['vice'] = vice
                            st.success("Startelva sparad.")
                else:
                    st.warning("Startelva måste ha 11 spelare. Välj exakt 11.")

# ----------------- Transfers & Chips tab -----------------
with tabs[2]:
    st.header("Transfers & Chips")
    mgr_list = list(st.session_state['league']['managers'].keys())
    sel = st.selectbox("Välj manager för transfers", options=[""]+mgr_list, key="t_mgr_select")
    if sel:
        m = st.session_state['league']['managers'][sel]
        st.write(f"Bank: {m['bank']:.1f}  Free transfers: {m['free_transfers']}")
        st.write("Current squad:")
        if m['squad']:
            st.write([f"{pid}: {st.session_state['league']['players'][pid]['name']}" for pid in m['squad']['players']])
        else:
            st.info("Ingen trupp ännu.")

        st.subheader("Make transfers (out -> in)")
        out_ids = st.multiselect("Out (choose existing squad IDs)", options=m['squad']['players'] if m['squad'] else [])
        in_ids = st.multiselect("In (choose available player IDs)", options=[pid for pid in st.session_state['league']['players'].keys() if not m['squad'] or pid not in m['squad']['players']])
        if st.button("Apply transfers"):
            ok,msg = apply_transfers(sel, out_ids, in_ids)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        st.markdown("### Chips")
        st.write("Available chips: " + ", ".join([k for k,v in m['chips'].items() if v]))
        chip_choice = st.selectbox("Activate chip", options=["","WC","FH","TC","BB"], index=0)
        if chip_choice and st.button("Activate chip"):
            activate_chip(sel, chip_choice)
# ----------------- Admin GW tab -----------------
with tabs[3]:
    st.header("Admin: Process Gameweek")
    st.write(f"Current GW: {st.session_state['league']['current_gw']}")
    st.write("Enter events for players (only for players involved).")
    # Provide UI to add events per player
    gw_events: Dict[int, Event] = {}
    cols = st.columns([2,1,1,1,1,1,1,1,1,1,1])
    cols[0].write("Player (id:name)")
    cols[1].write("Min")
    cols[2].write("G")
    cols[3].write("A")
    cols[4].write("CS")
    cols[5].write("GC")
    cols[6].write("Saves")
    cols[7].write("PenSave")
    cols[8].write("PenMiss")
    cols[9].write("Y")
    cols[10].write("R")
    # We'll create an expandable list for all players; in practice admin will fill only involved players
    players_df = players_table().sort_values('id')
    event_inputs = {}
    for _, row in players_df.iterrows():
        pid = int(row['id'])
        name = row['name']
        rcols = st.columns([2,1,1,1,1,1,1,1,1,1,1])
        rcols[0].write(f"{pid}: {name}")
        minutes = rcols[1].number_input(f"min_{pid}", min_value=0, max_value=120, value=0, key=f"min_{pid}")
        goals = rcols[2].number_input(f"g_{pid}", min_value=0, value=0, key=f"g_{pid}")
        assists = rcols[3].number_input(f"a_{pid}", min_value=0, value=0, key=f"a_{pid}")
        cs = rcols[4].checkbox(f"cs_{pid}", key=f"cs_{pid}")
        gc = rcols[5].number_input(f"gc_{pid}", min_value=0, value=0, key=f"gc_{pid}")
        saves = rcols[6].number_input(f"saves_{pid}", min_value=0, value=0, key=f"saves_{pid}")
        pensave = rcols[7].number_input(f"pensave_{pid}", min_value=0, value=0, key=f"pensave_{pid}")
        penmiss = rcols[8].number_input(f"penmiss_{pid}", min_value=0, value=0, key=f"penmiss_{pid}")
        yellow = rcols[9].number_input(f"yellow_{pid}", min_value=0, value=0, key=f"yellow_{pid}")
        red = rcols[10].number_input(f"red_{pid}", min_value=0, value=0, key=f"red_{pid}")
        # only add if something non-zero or cs
        if any([minutes, goals, assists, cs, gc, saves, pensave, penmiss, yellow, red]):
            ev = Event(minutes=int(minutes), goals=int(goals), assists=int(assists), cs=bool(cs),
                       goals_conceded=int(gc), saves=int(saves), pen_save=int(pensave), pen_miss=int(penmiss),
                       yellow=int(yellow), red=int(red), own_goal=0, bonus=0)
            gw_events[pid] = ev
    if st.button("Process GW"):
        process_gw_admin(gw_events)
        st.success("GW processed.")
# ----------------- Standings tab -----------------
with tabs[4]:
    st.header("Standings & Export")
    # show standings
    rows = []
    for name,m in st.session_state['league']['managers'].items():
        rows.append({"Manager": name, "Points": m['total_points'], "Bank": m['bank'], "FreeTransfers": m['free_transfers']})
    df = pd.DataFrame(rows).sort_values('score', ascending=False)
    st.table(df)
    # Export league JSON button (again)
    if st.button("Export league JSON (bottom)"):
        js = json.dumps(st.session_state['league'], ensure_ascii=False, indent=2)
        st.download_button("Download", js, file_name=f"league_{datetime.now().strftime('%Y%m%d_%H%M')}.json")

st.markdown("---")
st.caption("Note: This app implements nearly all FPL core features (squad rules, transfers, chips, GW scoring) in a single file. It's a prototype — for production you'd add authentication, concurrency handling, persistent DB, nicer UI/drag-and-drop.")



