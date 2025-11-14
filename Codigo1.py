# file: cleaning_mesa.py
# Run: python cleaning_mesa.py
# Then open: http://127.0.0.1:8521

import random
from typing import Tuple, Dict, List, Optional

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

# ---------- Domain ----------
Coord = Tuple[int, int]
DIRS = {
    "up":    (0, -1),
    "down":  (0, 1),
    "left":  (-1, 0),
    "right": (1, 0),
}
DIR_ORDER = ["up", "down", "left", "right"]


# ---------- Agents ----------
class Obstacle(Agent):
    """Static obstacle that blocks movement."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class Dirt(Agent):
    """Dirt with a type and value. Cleaned when the cleaner is on the cell and chooses 'clean'."""
    def __init__(self, unique_id, model, kind: str, value: int):
        super().__init__(unique_id, model)
        self.kind = kind
        self.value = value


class MemoryCleaner(Agent):
    """
    Cleaner with:
    - visited memory
    - value-driven neighbor selection
    - obstacle avoidance
    Decision policy:
      1) Clean if here dirty.
      2) Move to adjacent highest-value dirt.
      3) Explore unvisited valid neighbor.
      4) Else any valid move.
      5) Stay if no move.
    """
    def __init__(self, unique_id, model, start_pos: Coord):
        super().__init__(unique_id, model)
        self.pos: Coord = start_pos
        self.visited: set[Coord] = {start_pos}
        self.collected_value: int = 0
        self.cleaned_cells: int = 0
        self.last_action: str = "init"

    # ---- Perception helpers ----
    def _neighbors_info(self) -> List[Tuple[str, Coord, int]]:
        info = []
        for d in DIR_ORDER:
            dx, dy = DIRS[d]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if not self.model.grid.out_of_bounds((nx, ny)):
                # blocked by obstacle?
                blocked = any(isinstance(a, Obstacle) for a in self.model.grid.get_cell_list_contents((nx, ny)))
                if not blocked:
                    # sum of values if multiple dirts in the same cell (rare but supported)
                    v = 0
                    for a in self.model.grid.get_cell_list_contents((nx, ny)):
                        if isinstance(a, Dirt):
                            v += a.value
                    info.append((d, (nx, ny), v))
        return info

    def _here_dirty_value(self) -> int:
        v = 0
        for a in self.model.grid.get_cell_list_contents(self.pos):
            if isinstance(a, Dirt):
                v += a.value
        return v

    # ---- Decision ----
    def decide(self) -> str:
        here_v = self._here_dirty_value()
        if here_v > 0:
            return "clean"

        neighbors = self._neighbors_info()

        dirt_neighbors = [(d, p, v) for d, p, v in neighbors if v > 0]
        if dirt_neighbors:
            dirt_neighbors.sort(key=lambda t: (-t[2], DIR_ORDER.index(t[0])))
            return dirt_neighbors[0][0]

        unvisited = [(d, p) for d, p, v in neighbors if p not in self.visited]
        if unvisited:
            unvisited.sort(key=lambda t: DIR_ORDER.index(t[0]))
            return unvisited[0][0]

        if neighbors:
            neighbors.sort(key=lambda t: DIR_ORDER.index(t[0]))
            return neighbors[0][0]

        return "stay"

    # ---- Act ----
    def step(self):
        action = self.decide()
        self.last_action = action

        if action == "clean":
            # remove all dirts in the current cell, accumulate values
            cell_agents = list(self.model.grid.get_cell_list_contents(self.pos))
            gained = 0
            for a in cell_agents:
                if isinstance(a, Dirt):
                    gained += a.value
                    self.model.grid.remove_agent(a)
                    self.model.dirt_count -= 1
            if gained > 0:
                self.collected_value += gained
                self.cleaned_cells += 1
        elif action in DIRS:
            dx, dy = DIRS[action]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            # safe move (neighbors list already excludes obstacles/out-of-bounds)
            self.model.grid.move_agent(self, (nx, ny))
            self.pos = (nx, ny)
            self.visited.add(self.pos)
        else:
            # stay
            pass


# ---------- Model ----------
class CleaningModel(Model):
    """
    ABM with:
    - MultiGrid and RandomActivation
    - Multiple Dirt agents with types/values
    - Static Obstacle agents
    - One MemoryCleaner agent
    DataCollector tracks: remaining dirt, collected value, cleaned cells, step
    """
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        num_dirt: int = 18,
        num_obstacles: int = 15,
        seed: Optional[int] = 42
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.random = random.Random(seed)

        # dirt catalog
        self.dirt_catalog: Dict[str, Dict] = {
            "dust":   {"value": 1, "color": "#9aa0a6"},
            "crumb":  {"value": 2, "color": "#4285f4"},
            "spill":  {"value": 3, "color": "#ea4335"},
        }

        # place obstacles
        self.obstacle_ids: List[int] = []
        placed = 0
        while placed < num_obstacles:
            x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            if any(isinstance(a, Obstacle) for a in self.grid.get_cell_list_contents((x, y))):
                continue
            a = Obstacle(self.next_id(), self)
            self.grid.place_agent(a, (x, y))
            self.obstacle_ids.append(a.unique_id)
            placed += 1

        # place dirt (avoid obstacle cells)
        self.dirt_ids: List[int] = []
        self.dirt_count: int = 0
        placed = 0
        while placed < num_dirt:
            x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            cell_agents = self.grid.get_cell_list_contents((x, y))
            if any(isinstance(a, Obstacle) for a in cell_agents):
                continue
            kind = self.random.choice(list(self.dirt_catalog.keys()))
            val = self.dirt_catalog[kind]["value"]
            d = Dirt(self.next_id(), self, kind=kind, value=val)
            self.grid.place_agent(d, (x, y))
            self.dirt_ids.append(d.unique_id)
            self.dirt_count += 1
            placed += 1

        # place cleaner on a free cell
        while True:
            sx, sy = self.random.randrange(self.width), self.random.randrange(self.height)
            if not any(isinstance(a, Obstacle) for a in self.grid.get_cell_list_contents((sx, sy))):
                break
        self.cleaner = MemoryCleaner(self.next_id(), self, start_pos=(sx, sy))
        self.grid.place_agent(self.cleaner, (sx, sy))
        self.schedule.add(self.cleaner)

        self.step_count = 0
        self.datacollector = DataCollector(
            model_reporters={
                "remaining_dirt": lambda m: m.dirt_count,
                "collected_value": lambda m: m.cleaner.collected_value,
                "cleaned_cells": lambda m: m.cleaner.cleaned_cells,
                "step": lambda m: m.step_count,
            }
        )

    def step(self):
        self.schedule.step()
        self.step_count += 1
        self.datacollector.collect(self)

    # stopping rule example (optional)
    def all_clean(self) -> bool:
        return self.dirt_count == 0


# ---------- Visualization ----------
def portrayal(agent: Agent) -> Dict:
    if agent is None:
        return {}

    # base portrayal
    p = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}

    if isinstance(agent, Obstacle):
        p.update({"Color": "#202124"})  # dark gray
    elif isinstance(agent, Dirt):
        # color by type
        # we cannot access model.dirt_catalog directly here safely, so simple mapping
        color_map = {
            "dust": "#9aa0a6",
            "crumb": "#4285f4",
            "spill": "#ea4335",
        }
        p.update({"Color": color_map.get(agent.kind, "#9aa0a6"), "Layer": 1})
    elif isinstance(agent, MemoryCleaner):
        p.update({"Shape": "circle", "r": 0.5, "Color": "#34a853", "Layer": 2})
    else:
        p.update({"Color": "#ffffff"})

    return p


def run_server(width=10, height=10, num_dirt=18, num_obstacles=15, seed=42):
    grid_vis = CanvasGrid(portrayal, width, height, 600, 600)

    chart = ChartModule(
        [
            {"Label": "remaining_dirt", "Color": "black"},
            {"Label": "collected_value", "Color": "blue"},
            {"Label": "cleaned_cells", "Color": "red"},
        ],
        data_collector_name="datacollector",
    )

    server = ModularServer(
        CleaningModel,
        [grid_vis, chart],
        "Cleaning ABM: Memory + Multi-dirt + Obstacles",
        {
            "width": width,
            "height": height,
            "num_dirt": num_dirt,
            "num_obstacles": num_obstacles,
            "seed": seed,
        },
    )
    server.port = 8521
    server.launch()


if __name__ == "__main__":
    run_server()
