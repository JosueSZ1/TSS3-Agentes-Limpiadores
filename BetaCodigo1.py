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
    "up":    (0, 1),
    "down":  (0, -1),
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
    Exploración con memoria suave:
      1) limpiar si hay suciedad aquí
      2) ir al vecino con mayor valor (tie -> DIR_ORDER)
      3) explorar el vecino con MENOR visit_count (tie -> DIR_ORDER)
      4) si detecta bucle/estancamiento, romper pasando por el MÁS visitado
      5) si no hay vecinos válidos -> stay
    """
    def __init__(self, unique_id, model, start_pos):
        super().__init__(unique_id, model)
        self.pos = start_pos
        self.visited = {start_pos}
        self.visit_count = {start_pos: 1}
        self.collected_value = 0
        self.cleaned_cells = 0
        self.last_action = "init"

        # Para detección de bucles locales
        self.history = [start_pos]
        self.max_hist = 12              # ventana de historia
        self.loop_unique_threshold = 4  # si únicos <= 4 en la ventana => bucle probable

    # --------- helpers de percepción ----------
    def _neighbors_info(self):
        info = []
        for d in DIR_ORDER:
            dx, dy = DIRS[d]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if self.model.grid.out_of_bounds((nx, ny)):
                continue
            # bloqueado?
            if any(isinstance(a, Obstacle)
                   for a in self.model.grid.get_cell_list_contents((nx, ny))):
                continue
            # valor total de suciedad en el vecino
            v = 0
            for a in self.model.grid.get_cell_list_contents((nx, ny)):
                if isinstance(a, Dirt):
                    v += a.value
            info.append((d, (nx, ny), v))
        return info

    def _here_dirty_value(self):
        v = 0
        for a in self.model.grid.get_cell_list_contents(self.pos):
            if isinstance(a, Dirt):
                v += a.value
        return v

    def _looping(self, candidate_next_pos=None):
        """
        Heurística simple de bucle:
        - pocos únicos en la ventana
        - o el siguiente movimiento vuelve a una de las últimas 4 posiciones
        """
        if len(self.history) >= self.max_hist and len(set(self.history[-self.max_hist:])) <= self.loop_unique_threshold:
            return True
        if candidate_next_pos is not None:
            # volver una y otra vez a las mismas celdas cercanas
            recent = set(self.history[-4:])
            if candidate_next_pos in recent:
                return True
        return False

    # --------- decisión ----------
    def decide(self):
        # 1) limpiar si hay
        if self._here_dirty_value() > 0:
            return "clean"

        neighbors = self._neighbors_info()
        if not neighbors:
            return "stay"

        # 2) priorizar suciedad adyacente de mayor valor
        dirt_neighbors = [(d, p, v) for d, p, v in neighbors if v > 0]
        if dirt_neighbors:
            dirt_neighbors.sort(key=lambda t: (-t[2], DIR_ORDER.index(t[0])))
            best_dir, best_pos, _ = dirt_neighbors[0]
            # si limpiar no es posible ahora (no hay), pero vamos a movernos, revisa bucle
            if self._looping(candidate_next_pos=best_pos):
                # rompe el ciclo pasando por el más visitado
                return self._most_visited_dir(neighbors)
            return best_dir

        # 3) exploración least-visited
        def score(nb):
            d, p, v = nb
            return (self.visit_count.get(p, 0), DIR_ORDER.index(d))  # menor es mejor

        neighbors.sort(key=score)
        best_dir, best_pos, _ = neighbors[0]

        # 4) anti-bucle: si vamos a ciclar, escoge la alternativa más visitada
        if self._looping(candidate_next_pos=best_pos):
            return self._most_visited_dir(neighbors)

        return best_dir

    def _most_visited_dir(self, neighbors):
        """
        Fallback para romper atascos:
        elige el vecino con MAYOR visit_count (tie -> DIR_ORDER).
        """
        neighbors_sorted = sorted(
            neighbors,
            key=lambda nb: (-self.visit_count.get(nb[1], 0), DIR_ORDER.index(nb[0]))
        )
        return neighbors_sorted[0][0]

    # --------- actuar ----------
    def step(self):
        action = self.decide()
        self.last_action = action

        if action == "clean":
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
            self.model.grid.move_agent(self, (nx, ny))
            self.pos = (nx, ny)

            # memoria suave
            self.visited.add(self.pos)
            self.visit_count[self.pos] = self.visit_count.get(self.pos, 0) + 1
            self.history.append(self.pos)
            if len(self.history) > self.max_hist:
                self.history.pop(0)
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
                "suciedad_restante": lambda m: m.dirt_count,
                "valor_recogido": lambda m: m.cleaner.collected_value,
                "celdas_limpiadas": lambda m: m.cleaner.cleaned_cells,
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
            {"Label": "suciedad_restante", "Color": "black"},
            {"Label": "valor_recogido", "Color": "blue"},
            {"Label": "celdas_limpiadas", "Color": "red"},
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
