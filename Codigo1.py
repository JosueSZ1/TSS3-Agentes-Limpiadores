import random
from typing import Tuple, Dict, List, Optional

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

# ---------- Dominio ----------
Coord = Tuple[int, int]
DIRECCIONES = {
    "arriba":    (0, 1),
    "abajo":     (0, -1),
    "izquierda": (-1, 0),
    "derecha":   (1, 0),
}
ORDEN_DIRECCIONES = ["arriba", "abajo", "izquierda", "derecha"]

# ---------- Agentes ----------
class Obstaculo(Agent):
    """Obstáculo estático que bloquea el movimiento."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Suciedad(Agent):
    """Suciedad con tipo y valor. Se limpia cuando el limpiador está en la celda y decide 'limpiar'."""
    def __init__(self, unique_id, model, tipo: str, valor: int):
        super().__init__(unique_id, model)
        self.tipo = tipo
        self.valor = valor

class LimpiadorConMemoria(Agent):
    """
    Exploración con memoria suave:
      1) limpiar si hay suciedad aquí
      2) ir al vecino con mayor valor (empate -> ORDEN_DIRECCIONES)
      3) explorar el vecino con MENOR número de visitas (empate -> ORDEN_DIRECCIONES)
      4) si detecta bucle, pasar por el MÁS visitado
      5) si no hay vecinos válidos -> quedarse
    """
    def __init__(self, unique_id, model, start_pos):
        super().__init__(unique_id, model)
        self.pos = start_pos
        self.visited = {start_pos}
        self.visit_count = {start_pos: 1}
        self.valor_recogido = 0
        self.celdas_limpiadas = 0
        self.ultima_accion = "init"

        # Para detección de bucles locales
        self.historial = [start_pos]
        self.max_historial = 12               # ventana de historia
        self.umbral_bucle_unico = 4           # si hay <= 4 posiciones únicas en la ventana => bucle probable

    def _informacion_vecinos(self):
        """Obtiene información de los vecinos: dirección, posición, valor de suciedad."""
        info = []
        for d in ORDEN_DIRECCIONES:
            dx, dy = DIRECCIONES[d]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if self.model.grid.out_of_bounds((nx, ny)):
                continue
            # ¿Está bloqueado por un obstáculo?
            if any(isinstance(a, Obstaculo) for a in self.model.grid.get_cell_list_contents((nx, ny))):
                continue
            # Sumar el valor total de suciedad en el vecino
            valor_suciedad = 0
            for a in self.model.grid.get_cell_list_contents((nx, ny)):
                if isinstance(a, Suciedad):
                    valor_suciedad += a.valor
            info.append((d, (nx, ny), valor_suciedad))
        return info

    def _valor_suciedad_aqui(self):
        """Obtenemos el valor total de suciedad en la celda actual."""
        valor = 0
        for a in self.model.grid.get_cell_list_contents(self.pos):
            if isinstance(a, Suciedad):
                valor += a.valor
        return valor

    def _es_bucle(self, candidate_next_pos=None):
        """
        Heurística simple de bucle:
        - Pocos valores únicos en la ventana histórica
        - O el siguiente movimiento vuelve a una de las últimas 4 posiciones
        """
        if len(self.historial) >= self.max_historial and len(set(self.historial[-self.max_historial:])) <= self.umbral_bucle_unico:
            return True
        if candidate_next_pos is not None:
            recientes = set(self.historial[-4:])
            if candidate_next_pos in recientes:
                return True
        return False

    def decidir(self):
        """Lógica de decisión del limpiador."""
        # 1) Limpiar si hay suciedad en la celda actual
        if self._valor_suciedad_aqui() > 0:
            return "limpiar"

        vecinos = self._informacion_vecinos()
        if not vecinos:
            return "quedarse"

        # 2) Priorizar la suciedad adyacente con mayor valor
        vecinos_con_suciedad = [(d, p, v) for d, p, v in vecinos if v > 0]
        if vecinos_con_suciedad:
            vecinos_con_suciedad.sort(key=lambda t: (-t[2], ORDEN_DIRECCIONES.index(t[0])))
            mejor_direccion, mejor_pos, _ = vecinos_con_suciedad[0]
            if self._es_bucle(candidate_next_pos=mejor_pos):
                return self._direccion_mas_visitada(vecinos)
            return mejor_direccion

        # 3) Explorar el vecino menos visitado
        vecinos.sort(key=lambda nb: (self.visit_count.get(nb[1], 0), ORDEN_DIRECCIONES.index(nb[0])))
        mejor_direccion, mejor_pos, _ = vecinos[0]

        # 4) Si se detecta bucle, se elige la alternativa más visitada
        if self._es_bucle(candidate_next_pos=mejor_pos):
            return self._direccion_mas_visitada(vecinos)

        return mejor_direccion

    def _direccion_mas_visitada(self, vecinos):
        """
        Romper atascos: elegir el vecino con mayor visit_count (en caso de empate, usar ORDEN_DIRECCIONES).
        """
        vecinos_ordenados = sorted(
            vecinos,
            key=lambda nb: (-self.visit_count.get(nb[1], 0), ORDEN_DIRECCIONES.index(nb[0]))
        )
        return vecinos_ordenados[0][0]

    def step(self):
        """Realiza un paso de simulación."""
        accion = self.decidir()
        self.ultima_accion = accion

        if accion == "limpiar":
            agentes_en_celda = list(self.model.grid.get_cell_list_contents(self.pos))
            ganado = 0
            for a in agentes_en_celda:
                if isinstance(a, Suciedad):
                    ganado += a.valor
                    self.model.grid.remove_agent(a)
                    self.model.dirt_count -= 1
            if ganado > 0:
                self.valor_recogido += ganado
                self.celdas_limpiadas += 1
        elif accion in DIRECCIONES:
            dx, dy = DIRECCIONES[accion]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            self.model.grid.move_agent(self, (nx, ny))
            self.pos = (nx, ny)

            # Memoria suave
            self.visited.add(self.pos)
            self.visit_count[self.pos] = self.visit_count.get(self.pos, 0) + 1
            self.historial.append(self.pos)
            if len(self.historial) > self.max_historial:
                self.historial.pop(0)
        else:
            pass

# ---------- Modelo ----------
class ModeloLimpieza(Model):
    """
    ABM con:
    - MultiGrid y RandomActivation
    - Varios agentes de Suciedad con tipos y valores
    - Agentes Obstáculo estáticos
    - Un agente Limpiador con Memoria
    - DataCollector para seguir el progreso
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

        # Catálogo de suciedad
        self.catalogo_suciedad: Dict[str, Dict] = {
            "polvo":   {"valor": 1, "color": "#9aa0a6"},
            "migaja":  {"valor": 2, "color": "#4285f4"},
            "derrame": {"valor": 3, "color": "#ea4335"},
        }

        # Colocando obstáculos
        self.ids_obstaculos: List[int] = []
        colocado = 0
        while colocado < num_obstacles:
            x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            if any(isinstance(a, Obstaculo) for a in self.grid.get_cell_list_contents((x, y))):
                continue
            a = Obstaculo(self.next_id(), self)
            self.grid.place_agent(a, (x, y))
            self.ids_obstaculos.append(a.unique_id)
            colocado += 1

        # Colocando suciedad (evitando obstáculos)
        self.ids_suciedad: List[int] = []
        self.dirt_count: int = 0
        colocado = 0
        while colocado < num_dirt:
            x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            agentes_celda = self.grid.get_cell_list_contents((x, y))
            if any(isinstance(a, Obstaculo) for a in agentes_celda):
                continue
            tipo = self.random.choice(list(self.catalogo_suciedad.keys()))
            valor = self.catalogo_suciedad[tipo]["valor"]
            s = Suciedad(self.next_id(), self, tipo=tipo, valor=valor)
            self.grid.place_agent(s, (x, y))
            self.ids_suciedad.append(s.unique_id)
            self.dirt_count += 1
            colocado += 1

        # Colocando el limpiador en una celda libre
        while True:
            sx, sy = self.random.randrange(self.width), self.random.randrange(self.height)
            if not any(isinstance(a, Obstaculo) for a in self.grid.get_cell_list_contents((sx, sy))):
                break
        self.limpiador = LimpiadorConMemoria(self.next_id(), self, start_pos=(sx, sy))
        self.grid.place_agent(self.limpiador, (sx, sy))
        self.schedule.add(self.limpiador)

        self.step_count = 0
        self.datacollector = DataCollector(
            model_reporters={
                "suciedad_restante": lambda m: m.dirt_count,
                "valor_recogido": lambda m: m.limpiador.valor_recogido,
                "celdas_limpiadas": lambda m: m.limpiador.celdas_limpiadas,
                "paso": lambda m: m.step_count,
            }
        )

    def step(self):
        self.schedule.step()
        self.step_count += 1
        self.datacollector.collect(self)

    # Regla de parada (opcional)
    def todo_limpio(self) -> bool:
        return self.dirt_count == 0


# ---------- Visualización ----------
def presentación(agent: Agent) -> Dict:
    """Definición visual para cada agente en el grid."""
    if agent is None:
        return {}

    # visualización base
    p = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}

    if isinstance(agent, Obstaculo):
        p.update({"Color": "#202124"})  # gris oscuro
    elif isinstance(agent, Suciedad):
        # color según tipo
        color_map = {
            "polvo": "#9aa0a6",
            "migaja": "#4285f4",
            "derrame": "#ea4335",
        }
        p.update({"Color": color_map.get(agent.tipo, "#9aa0a6"), "Layer": 1})
    elif isinstance(agent, LimpiadorConMemoria):
        p.update({"Shape": "circle", "r": 0.5, "Color": "#34a853", "Layer": 2})
    else:
        p.update({"Color": "#ffffff"})

    return p


def correr_servidor(width=10, height=10, num_dirt=15, num_obstacles=15, seed=42):
    grid_vis = CanvasGrid(presentación, width, height, 600, 600)

    chart = ChartModule(
        [
            {"Label": "suciedad_restante", "Color": "black"},
            {"Label": "valor_recogido", "Color": "blue"},
            {"Label": "celdas_limpiadas", "Color": "red"},
        ],
        data_collector_name="datacollector",
    )

    servidor = ModularServer(
        ModeloLimpieza,
        [grid_vis, chart],
        "Modelo de Limpieza ABM: Memoria + Suciedad + Obstáculos",
        {
            "width": width,
            "height": height,
            "num_dirt": num_dirt,
            "num_obstacles": num_obstacles,
            "seed": seed,
        },
    )
    servidor.port = 8521
    servidor.launch()


if __name__ == "__main__":
    correr_servidor()
