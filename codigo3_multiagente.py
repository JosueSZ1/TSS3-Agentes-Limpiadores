import random

class CooperativeAgent:
    """Agent that can communicate simple messages to others."""

    def __init__(self, agent_id: int, x: int, y: int, env: "MultiAgentEnvironment"):
        self.id = agent_id
        self.x = x
        self.y = y
        self.env = env
        self.collected_food = 0
        self.target = None
        self.inbox = []  # received messages

    # --- Communication ---
    def send_message(self, recipients, kind, payload):
        for agent in recipients:
            agent.receive_message(self.id, kind, payload)

    def receive_message(self, sender_id, kind, payload):
        self.inbox.append({"from": sender_id, "type": kind, "payload": payload})

    def process_messages(self):
        reported_food = []
        for msg in self.inbox:
            if msg["type"] == "food_found":
                reported_food.append(msg["payload"])
        self.inbox.clear()
        return reported_food

    # --- Perception & Action ---
    def perceive(self):
        return self.env.get_nearby_food(self.x, self.y, radius=3)

    def decide_and_act(self, others):
        # Process communications
        shared_food = self.process_messages()

        # Local perception
        local_food = self.perceive()

        # Share local findings
        if local_food and others:
            for pos in local_food:
                self.send_message(others, "food_found", pos)

        # Choose or maintain target
        options = list(set(local_food + shared_food))
        if options and self.target is None:
            self.target = min(options, key=lambda p: abs(p[0] - self.x) + abs(p[1] - self.y))

        # Move towards target
        if self.target:
            if (self.x, self.y) == self.target:
                if self.env.collect_food(self.x, self.y):
                    self.collected_food += 1
                self.target = None
            else:
                dx = 0
                dy = 0
                if self.target[0] > self.x:
                    dx = 1
                elif self.target[0] < self.x:
                    dx = -1
                elif self.target[1] > self.y:
                    dy = 1
                elif self.target[1] < self.y:
                    dy = -1

                nx, ny = self.x + dx, self.y + dy
                if self.env.is_valid(nx, ny):
                    self.x, self.y = nx, ny
        else:
            # Random walk
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            nx, ny = self.x + dx, self.y + dy
            if self.env.is_valid(nx, ny):
                self.x, self.y = nx, ny


class MultiAgentEnvironment:
    """Grid with food for multiple agents."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.food = set()
        for _ in range(15):
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            self.food.add((x, y))

    def is_valid(self, x, y) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_nearby_food(self, x, y, radius: int):
        return [pos for pos in self.food if abs(pos[0] - x) + abs(pos[1] - y) <= radius]

    def collect_food(self, x, y) -> bool:
        if (x, y) in self.food:
            self.food.remove((x, y))
            return True
        return False

    def show(self, agents):
        grid = [["⬜" for _ in range(self.width)] for _ in range(self.height)]
        for fx, fy in self.food:
            grid[fy][fx] = "🍎"
        for a in agents:
            grid[a.y][a.x] = str(a.id)
        for row in grid:
            print(" ".join(row))
        print()


def simulate_multi_agent(num_agents: int = 3, steps: int = 25):
    env = MultiAgentEnvironment(10, 10)
    agents = []
    for i in range(num_agents):
        x, y = random.randint(0, 9), random.randint(0, 9)
        agents.append(CooperativeAgent(i + 1, x, y, env))

    print("=== SIMULATION: COOPERATIVE MULTI-AGENT SYSTEM ===\n")
    print("Initial state:")
    env.show(agents)

    for step in range(steps):
        for agent in agents:
            others = [a for a in agents if a.id != agent.id]
            agent.decide_and_act(others)

        if step % 5 == 0:
            print(f"\nStep {step + 1}:")
            env.show(agents)
            for agent in agents:
                print(f"Agent {agent.id}: {agent.collected_food} food")

        if len(env.food) == 0:
            print("\nAll food has been collected!")
            break

    print("\nFinal result:")
    total = sum(a.collected_food for a in agents)
    for agent in agents:
        print(f"Agent {agent.id}: {agent.collected_food} food")
    print(f"Total collected: {total}")


if __name__ == "__main__":
    simulate_multi_agent()
