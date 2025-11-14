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

class Comida(Agent):
    """Comida en una celda, que puede ser recogida por los agentes."""
    def __init__(self, unique_id, model, valor: int):
        super().__init__(unique_id, model)
        self.valor = valor

class AgenteConMemoria(Agent):
    """
    Agente que explora el entorno con memoria:
      1) Recoge comida si la encuentra
      2) Explora el vecino con más comida (memoria espacial)
      3) Si está atrapado, se mueve heurísticamente para salir del bucle
    """
    def __init__(self, unique_id, model, start_pos):
        super().__init__(unique_id, model)
        self.pos = start_pos
        self.visited = {start_pos}  # Registra las posiciones visitadas
        self.comida_recogida = 0
        self.historial = [start_pos]  # Historial de las posiciones visitadas para detectar bucles
        self.max_historial = 12  # Número máximo de pasos en el historial
        self.umbral_bucle_unico = 4  # Umbral para identificar un bucle (número de posiciones únicas)
        self.visit_count = {start_pos: 1}  # Cuenta las visitas a cada celda

    def _informacion_vecinos(self):
        """Obtiene información sobre los vecinos: dirección, posición, comida disponible."""
        info = []
        for d in ORDEN_DIRECCIONES:
            dx, dy = DIRECCIONES[d]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if self.model.grid.out_of_bounds((nx, ny)):
                continue
            if any(isinstance(a, Obstaculo) for a in self.model.grid.get_cell_list_contents((nx, ny))):
                continue  # No moverse si hay un obstáculo en la celda
            comida_valor = sum(a.valor for a in self.model.grid.get_cell_list_contents((nx, ny)) if isinstance(a, Comida))
            info.append((d, (nx, ny), comida_valor))
        return info

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

    def _direccion_mas_visitada(self, vecinos):
        """
        Romper atascos: elegir el vecino con mayor visit_count (en caso de empate, usar ORDEN_DIRECCIONES).
        """
        # Ordena los vecinos por el número de veces que fueron visitados (más visitados primero)
        vecinos_ordenados = sorted(
            vecinos,
            key=lambda nb: (-self.visit_count.get(nb[1], 0), ORDEN_DIRECCIONES.index(nb[0]))
        )
        return vecinos_ordenados[0][0]

    def decidir(self):
        """Decisión del agente para moverse hacia el área con más comida y salir de un bucle si es necesario."""
        # 1) Recoger comida si la encuentra en la celda actual
        if self._valor_comida_aqui() > 0:
            return "limpiar"  # Se llama "limpiar" porque estamos recolectando comida

        vecinos = self._informacion_vecinos()
        if not vecinos:
            return "quedarse"  # Si no hay vecinos válidos, el agente se queda

        # 2) Priorizar los vecinos con más comida
        vecinos_con_comida = [(d, p, v) for d, p, v in vecinos if v > 0]
        if vecinos_con_comida:
            # Ordena los vecinos con comida por el valor de la comida (mayor primero)
            vecinos_con_comida.sort(key=lambda t: (-t[2], ORDEN_DIRECCIONES.index(t[0])))
            mejor_direccion, mejor_pos, _ = vecinos_con_comida[0]

            # Si está en un bucle, mover al vecino con el menor peso
            if self._es_bucle(candidate_next_pos=mejor_pos):
                return self._direccion_mas_visitada(vecinos)
            return mejor_direccion

        # 3) Si no hay comida visible, moverse hacia el vecino menos visitado
        vecinos.sort(key=lambda nb: (self.visit_count.get(nb[1], 0), ORDEN_DIRECCIONES.index(nb[0])))
        mejor_direccion, mejor_pos, _ = vecinos[0]

        # 4) Si se detecta un bucle en la dirección seleccionada, moverse al vecino más visitado
        if self._es_bucle(candidate_next_pos=mejor_pos):
            return self._direccion_mas_visitada(vecinos)

        return mejor_direccion

    def _valor_comida_aqui(self):
        """Obtenemos el valor total de comida en la celda actual."""
        valor = 0
        for a in self.model.grid.get_cell_list_contents(self.pos):
            if isinstance(a, Comida):
                valor += a.valor
        return valor

    def step(self):
        """Realiza un paso de simulación."""
        accion = self.decidir()
        if accion == "quedarse":
            return

        dx, dy = DIRECCIONES[accion]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy

        # Verificar si el nuevo movimiento está dentro de los límites de la cuadrícula y no hay obstáculo
        if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
            # Asegurarse de que no hay un obstáculo en la nueva celda
            if not any(isinstance(a, Obstaculo) for a in self.model.grid.get_cell_list_contents((nx, ny))):
                self.model.grid.move_agent(self, (nx, ny))
                self.pos = (nx, ny)
            else:
                # Si hay un obstáculo, el agente no se mueve
                pass
        else:
            # Si está fuera de los límites, no hacer nada
            pass

        # Verificar si hay comida en la celda después de moverse
        agentes_en_celda = list(self.model.grid.get_cell_list_contents(self.pos))
        for a in agentes_en_celda:
            if isinstance(a, Comida):
                self.comida_recogida += a.valor
                self.model.grid.remove_agent(a)  # Elimina la comida de la celda
                self.model.comida_count -= 1     # Reduce el contador de comida
                break  # Solo recoge una vez por paso

        # Actualizar el historial de posiciones visitadas
        self.visited.add(self.pos)
        self.historial.append(self.pos)
        self.visit_count[self.pos] = self.visit_count.get(self.pos, 0) + 1

# ---------- Modelo ----------
class ModeloRecoleccion(Model):
    """
    ABM con:
    - MultiGrid y RandomActivation
    - Agentes de Comida y Obstáculos
    - Un agente recolector con Memoria
    """
    def __init__(self, width=10, height=10, num_comida=15, num_obstaculos=10, seed=42):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.random = random.Random(seed)

        # Crear comida
        self.comida_count = 0
        for _ in range(num_comida):
            x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            comida = Comida(self.next_id(), self, valor=random.randint(1, 3))
            self.grid.place_agent(comida, (x, y))
            self.comida_count += 1

        # Crear obstáculos
        for _ in range(num_obstaculos):
            x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            # Verificar que la celda esté libre de comida y obstáculos
            while any(isinstance(a, Obstaculo) for a in self.grid.get_cell_list_contents((x, y))) or \
                any(isinstance(a, Comida) for a in self.grid.get_cell_list_contents((x, y))):
                x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            
            obstaculo = Obstaculo(self.next_id(), self)
            self.grid.place_agent(obstaculo, (x, y))
        # Crear agente recolector
        sx, sy = self.random.randrange(self.width), self.random.randrange(self.height)
        self.recolector = AgenteConMemoria(self.next_id(), self, start_pos=(sx, sy))
        self.grid.place_agent(self.recolector, (sx, sy))
        self.schedule.add(self.recolector)

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={"comida_recogida": lambda m: m.recolector.comida_recogida}
        )

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def todo_recolectado(self):
        return self.comida_count == 0


# ---------- Visualización ----------
def presentacion(agent: Agent) -> Dict:
    """Definición visual para cada agente en el grid."""
    if agent is None:
        return {}

    # Agentes
    p = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}

    if isinstance(agent, Obstaculo):
        p.update({"Color": "#202124"})  # Obstáculo gris
    elif isinstance(agent, Comida):
        p.update({"Color": "#34a853", "Layer": 1})  # Comida verde
    elif isinstance(agent, AgenteConMemoria):
        p.update({"Shape": "circle", "r": 0.5, "Color": "#4285f4", "Layer": 2})  # Agente azul
    return p


# ---------- Servidor ----------
def correr_servidor(width=10, height=10, num_comida=15, num_obstaculos=10, seed=42):
    grid_vis = CanvasGrid(presentacion, width, height, 600, 600)

    chart = ChartModule(
        [{"Label": "comida_recogida", "Color": "blue"}],
        data_collector_name="datacollector",
    )

    servidor = ModularServer(
        ModeloRecoleccion,
        [grid_vis, chart],
        "Modelo de Recolección de Comida",
        {
            "width": width,
            "height": height,
            "num_comida": num_comida,
            "num_obstaculos": num_obstaculos,
            "seed": seed,
        },
    )
    servidor.port = 8521
    servidor.launch()


if __name__ == "__main__":
    correr_servidor()
