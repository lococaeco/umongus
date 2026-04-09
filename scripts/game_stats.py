#!/usr/bin/env python3
"""Print per-game and aggregate statistics for an AmongUs experiment directory.

Usage:
    python scripts/game_stats.py <expt-logs-dir>

Example:
    python scripts/game_stats.py expt-logs/2026-04-09_exp_3
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime


def parse_compact_jsonl(path):
    """The compact log is a stream of concatenated JSON objects (not newline-delimited)."""
    text = open(path).read()
    dec = json.JSONDecoder()
    i = 0
    objs = []
    while i < len(text):
        while i < len(text) and text[i] in " \n\r\t,":
            i += 1
        if i >= len(text):
            break
        obj, end = dec.raw_decode(text, i)
        objs.append(obj)
        i = end
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expt_dir", help="experiment directory with agent-logs-compact.json and summary.json")
    args = ap.parse_args()

    log_path = os.path.join(args.expt_dir, "agent-logs-compact.json")
    summary_path = os.path.join(args.expt_dir, "summary.json")
    if not os.path.exists(log_path):
        sys.exit(f"missing {log_path}")

    objs = parse_compact_jsonl(log_path)

    per_game = defaultdict(list)
    for o in objs:
        per_game[o["game_index"]].append(o)

    summary = {}
    if os.path.exists(summary_path):
        try:
            text = open(summary_path).read()
            dec = json.JSONDecoder()
            i = 0
            while i < len(text):
                while i < len(text) and text[i] in " \n\r\t,":
                    i += 1
                if i >= len(text):
                    break
                obj, end = dec.raw_decode(text, i)
                summary.update(obj)  # merge {Game N: {...}} fragments
                i = end
        except Exception as e:
            print(f"(warning) could not parse summary.json: {e}")

    rows = []
    for game, entries in sorted(per_game.items(), key=lambda kv: int(kv[0].split()[-1])):
        entries.sort(key=lambda o: o["timestamp"])
        t0 = datetime.fromisoformat(entries[0]["timestamp"])
        t1 = datetime.fromisoformat(entries[-1]["timestamp"])
        wall_seconds = (t1 - t0).total_seconds()

        max_step = max(o["step"] for o in entries)
        players = {o["player"]["name"]: o["player"]["identity"] for o in entries}
        impostors = [p for p, r in players.items() if r == "Impostor"]
        n_players = len(players)
        total_llm_calls = len(entries)

        game_summary = summary.get(game, {})
        winner_reason = game_summary.get("winner_reason", "")
        # winner_reason is the source of truth; winner-code meaning differs between
        # upstream builds (seen values so far: 1 = Impostors, 3 = Crewmates).
        if "Crewmates win" in winner_reason:
            winner = "Crewmates"
        elif "Impostors win" in winner_reason:
            winner = "Impostors"
        else:
            winner = "unknown"

        rows.append({
            "game": game,
            "winner": winner,
            "steps": max_step,
            "wall_s": wall_seconds,
            "llm_calls": total_llm_calls,
            "n_players": n_players,
            "impostors": impostors,
            "reason": winner_reason,
        })

    # Per-game table
    header = f"{'GAME':<10} {'WINNER':<10} {'STEPS':<6} {'WALL(s)':<9} {'LLM CALLS':<10} {'IMPOSTORS'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        imp_str = ", ".join(r["impostors"])
        print(f"{r['game']:<10} {r['winner']:<10} {r['steps']:<6} {r['wall_s']:<9.1f} {r['llm_calls']:<10} {imp_str}")

    # Aggregate
    if rows:
        print()
        print("=" * 60)
        print(f"Aggregate over {len(rows)} games")
        print("=" * 60)
        wins = Counter(r["winner"] for r in rows)
        print(f"  Win rate: Crewmates {wins.get('Crewmates', 0)}/{len(rows)}, Impostors {wins.get('Impostors', 0)}/{len(rows)}")
        avg_steps = sum(r["steps"] for r in rows) / len(rows)
        avg_wall = sum(r["wall_s"] for r in rows) / len(rows)
        avg_calls = sum(r["llm_calls"] for r in rows) / len(rows)
        print(f"  Avg steps per game: {avg_steps:.1f}")
        print(f"  Avg wall time per game: {avg_wall:.1f}s")
        print(f"  Avg LLM calls per game: {avg_calls:.1f}")
        print(f"  Total LLM calls: {sum(r['llm_calls'] for r in rows)}")


if __name__ == "__main__":
    main()
