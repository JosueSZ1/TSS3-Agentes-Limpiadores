import random
from typing import Tuple, Dict, List, Optional
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

# ---------- Dominio ----------
Coordenada = Tuple[int, int]
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
    """Recurso (comida) limitado que los agentes compiten por recolectar."""
    def __init__(self, unique_id, model, valor: int = 1):
        super().__init__(unique_id, model)
        self.valor = valor


class AgenteRecolector(Agent):
    """
    Agente recolector que:
      - Compite por comida limitada.
      - Se comunica con otros para evitar ir al mismo objetivo.

    Protocolo sencillo:
      * Comparte posiciones de comida que ve: mensaje tipo 'comida_encontrada'.
      * Cuando toma un objetivo, envía 'objetivo_tomado' con la posición.
      * Cada agente evita elegir objetivos que están marcados como tomados.
    """

    def __init__(self, unique_id, model, start_pos: Coordenada):
        super().__init__(unique_id, model)
        self.pos = start_pos
        self.objetivo: Optional[Coordenada] = None
        self.buzon: List[Dict] = []
        self.comida_recolectada: int = 0
        # Conjunto local de objetivos que se asume ya están tomados por alguien
        self.objetivos_reservados: set[Coordenada] = set()

    # --- Comunicación ---
    def enviar_mensaje(self, destinatarios: List["AgenteRecolector"], tipo: str, contenido):
        for agente in destinatarios:
            agente.recibir_mensaje(self.unique_id, tipo, contenido)

    def recibir_mensaje(self, remitente_id: int, tipo: str, contenido):
        self.buzon.append({"de": remitente_id, "tipo": tipo, "contenido": contenido})

    def procesar_mensajes(self) -> List[Coordenada]:
        """
        Procesa mensajes recibidos.
        Devuelve lista de posiciones de comida reportadas por otros agentes.
        Actualiza los objetivos reservados conocidos.
        """
        comida_reportada: List[Coordenada] = []
        nuevos_reservados: List[Coordenada] = []

        for msg in self.buzon:
            if msg["tipo"] == "comida_encontrada":
                comida_reportada.append(msg["contenido"])
            elif msg["tipo"] == "objetivo_tomado":
                nuevos_reservados.append(msg["contenido"])

        # Actualizar memoria de objetivos reservados
        self.objetivos_reservados.update(nuevos_reservados)
        self.buzon.clear()
        return comida_reportada

    # --- Percepción ---
    def percibir_comida(self, radio: int = 3) -> List[Coordenada]:
        """
        Devuelve posiciones de comida dentro de un radio de visión (distancia Manhattan).
        """
        return self.model.obtener_comida_cercana(self.pos, radio)

    # --- Decisión + Acción ---
    def elegir_objetivo(self, opciones: List[Coordenada]) -> Optional[Coordenada]:
        """
        Elige el objetivo más cercano que no esté reservado por otros agentes.
        """
        if not opciones:
            return None

        # Filtrar posiciones que ya están reservadas
        libres = [p for p in opciones if p not in self.objetivos_reservados]
        if not libres:
            return None

        # Elegir la comida más cercana (distancia Manhattan)
        x, y = self.pos
        return min(libres, key=lambda p: abs(p[0] - x) + abs(p[1] - y))

    def mover_hacia(self, destino: Coordenada):
        """Movimiento greedy en rejilla (una celda por paso) evitando obstáculos y bordes."""
        x, y = self.pos
        tx, ty = destino

        candidatos: List[Coordenada] = []

        # Priorizamos eje x, luego eje y (puedes ajustar esto si quieres)
        if tx > x:
            candidatos.append((x + 1, y))
        elif tx < x:
            candidatos.append((x - 1, y))
        if ty > y:
            candidatos.append((x, y + 1))
        elif ty < y:
            candidatos.append((x, y - 1))

        # Elegir el primer candidato válido y sin obstáculo
        for nx, ny in candidatos:
            if self.model.es_celda_libre((nx, ny)):
                self.model.grid.move_agent(self, (nx, ny))
                return

        # Si no puede avanzar hacia el objetivo, moverse aleatoriamente (si hay celda libre)
        dx, dy = random.choice(list(DIRECCIONES.values()))
        nx, ny = x + dx, y + dy
        if self.model.es_celda_libre((nx, ny)):
            self.model.grid.move_agent(self, (nx, ny))

    def step(self):
        """Llamado automáticamente por el scheduler de Mesa en cada tick."""
        # Otros agentes (para comunicarme)
        otros: List[AgenteRecolector] = [
            a for a in self.model.agentes if a.unique_id != self.unique_id
        ]

        # 1) Procesar mensajes
        comida_compartida = self.procesar_mensajes()

        # 2) Percibir comida local
        comida_local = self.percibir_comida()

        # 3) Compartir descubrimientos de comida local con otros
        if comida_local and otros:
            for pos in comida_local:
                self.enviar_mensaje(otros, "comida_encontrada", pos)

        # 4) Elegir nuevo objetivo si no tengo
        if self.objetivo is None:
            opciones = list(set(comida_local + comida_compartida))
            nuevo = self.elegir_objetivo(opciones)
            if nuevo is not None:
                self.objetivo = nuevo
                # Reservar objetivo localmente y avisar a los demás
                self.objetivos_reservados.add(self.objetivo)
                if otros:
                    self.enviar_mensaje(otros, "objetivo_tomado", self.objetivo)

        # 5) Actuar según el objetivo
        if self.objetivo is not None:
            # Si ya estoy en el objetivo, intento recolectar
            if self.pos == self.objetivo:
                if self.model.recolectar_comida(self.pos):
                    self.comida_recolectada += 1
                # Libero el objetivo (ya no tiene comida)
                self.objetivo = None
            else:
                # Moverme hacia el objetivo
                self.mover_hacia(self.objetivo)
        else:
            # Si no tengo objetivo, paseo aleatorio
            x, y = self.pos
            dx, dy = random.choice(list(DIRECCIONES.values()))
            nx, ny = x + dx, y + dy
            if self.model.es_celda_libre((nx, ny)):
                self.model.grid.move_agent(self, (nx, ny))


# ---------- Modelo ----------
class ModeloRecolectoresCompetitivos(Model):
    """
    Sistema multi-agente donde:
      - Hay recursos (Comida) LIMITADOS.
      - Varios agentes recolectores COMPITEN por ellos.
      - Se COORDINAN mediante comunicación para no ir al mismo objetivo.
    """

    def __init__(self, width: int = 10, height: int = 10, num_comida: int = 15, num_obstaculos: int = 10, num_agentes: int = 3, seed: Optional[int] = 42):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.random = random.Random(seed)

        self.agentes: List[AgenteRecolector] = []
        self.comida_restante: int = 0

        # ----- Colocar obstáculos -----
        for _ in range(num_obstaculos):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            if any(isinstance(a, Obstaculo) for a in self.grid.get_cell_list_contents((x, y))):
                continue
            ob = Obstaculo(self.next_id(), self)
            self.grid.place_agent(ob, (x, y))

        # ----- Colocar comida -----
        colocada = 0
        while colocada < num_comida:
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            if any(isinstance(a, Obstaculo) for a in self.grid.get_cell_list_contents((x, y))):
                continue
            comida = Comida(self.next_id(), self, valor=1)
            self.grid.place_agent(comida, (x, y))
            colocada += 1
            self.comida_restante += 1

        # ----- Colocar agentes recolectores -----
        for _ in range(num_agentes):
            while True:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                if any(isinstance(a, Obstaculo) for a in self.grid.get_cell_list_contents((x, y))):
                    continue
                agente = AgenteRecolector(self.next_id(), self, (x, y))
                self.grid.place_agent(agente, (x, y))
                self.schedule.add(agente)
                self.agentes.append(agente)
                break

        self.datacollector = DataCollector(
            model_reporters={
                "comida_restante": lambda m: m.comida_restante,
                "comida_recolectada_total": lambda m: sum(a.comida_recolectada for a in m.agentes),
            }
        )

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def es_dentro_de_grid(self, pos: Coordenada) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def es_celda_libre(self, pos: Coordenada) -> bool:
        """Celda válida y sin obstáculo (puede tener comida u otros agentes)."""
        if not self.es_dentro_de_grid(pos):
            return False
        contenido = self.grid.get_cell_list_contents(pos)
        return not any(isinstance(a, Obstaculo) for a in contenido)

    def obtener_comida_cercana(self, pos: Coordenada, radio: int) -> List[Coordenada]:
        """Devuelve posiciones de comida en un radio de distancia Manhattan."""
        x0, y0 = pos
        posiciones: List[Coordenada] = []
        for dx in range(-radio, radio + 1):
            for dy in range(-radio, radio + 1):
                nx, ny = x0 + dx, y0 + dy
                if not self.es_dentro_de_grid((nx, ny)):
                    continue
                if abs(dx) + abs(dy) > radio:
                    continue
                for ag in self.grid.get_cell_list_contents((nx, ny)):
                    if isinstance(ag, Comida):
                        posiciones.append((nx, ny))
                        break
        return posiciones

    def recolectar_comida(self, pos: Coordenada) -> bool:
        """Elimina una unidad de comida en la celda si existe."""
        contenido = self.grid.get_cell_list_contents(pos)
        for ag in contenido:
            if isinstance(ag, Comida):
                self.grid.remove_agent(ag)
                self.comida_restante -= 1
                return True
        return False


# ---------- Visualización ----------

def presentacion(agent: Agent) -> Dict:
    """Definición visual para cada agente en el grid."""
    if agent is None:
        return {}

    portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}

    if isinstance(agent, Obstaculo):
        portrayal.update({"Color": "#202124", "Layer": 0})
    elif isinstance(agent, Comida):
        portrayal.update({"Color": "#ffcc00", "Layer": 1})
    elif isinstance(agent, AgenteRecolector):
        portrayal.update({"Shape": "circle", "Color": "#34a853", "Layer": 2, "r": 0.6})
    return portrayal


def correr_servidor(width=10, height=10, num_comida=15, num_obstaculos=10, num_agentes=3, seed=42):
    """Lanza el servidor para la visualización de la simulación."""
    grid_vis = CanvasGrid(presentacion, width, height, 600, 600)
    chart = ChartModule(
        [
            {"Label": "comida_restante", "Color": "black"},
            {"Label": "comida_recolectada_total", "Color": "blue"},
        ],
        data_collector_name="datacollector",
    )

    servidor = ModularServer(
        ModeloRecolectoresCompetitivos,
        [grid_vis, chart],
        "Recolectores competitivos con comunicación",
        {
            "width": width,
            "height": height,
            "num_comida": num_comida,
            "num_obstaculos": num_obstaculos,
            "num_agentes": num_agentes,
            "seed": seed,
        },
    )
    servidor.port = 8521
    servidor.launch()


if __name__ == "__main__":
    correr_servidor()
