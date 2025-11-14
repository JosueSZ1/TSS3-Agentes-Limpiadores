import random
from collections import deque

class CollectorAgent:
    """Goal-based agent that plans routes to food using BFS."""

    def __init__(self, x: int, y: int, env: "GatheringEnvironment"):
        self.x = x
        self.y = y
        self.env = env
        self.energy = 100
        self.collected_food = 0
        self.plan = []  # Planned sequence of moves

    def perceive(self):
        """Return visible food positions within a vision radius."""
        return self.env.get_visible_food(self.x, self.y, radius=5)

    def plan_route(self, target):
        """Breadth-first search to find a path to target cell."""
        if target is None:
            return []

        queue = deque([(self.x, self.y, [])])
        visited = {(self.x, self.y)}

        while queue:
            x, y, path = queue.popleft()

            if (x, y) == target:
                return path

            for dx, dy, move in [(0, -1, "up"), (0, 1, "down"),
                                 (-1, 0, "left"), (1, 0, "right")]:
                nx, ny = x + dx, y + dy
                if (self.env.is_valid(nx, ny)
                    and (nx, ny) not in visited
                    and not self.env.has_obstacle(nx, ny)):
                    visited.add((nx, ny))
                    queue.append((nx, ny, path + [move]))
        return []  # No path found

    def decide(self, visible_food):
        """Choose target and produce next action."""
        if not self.plan and visible_food:
            target = min(visible_food, key=lambda c: abs(c[0] - self.x) + abs(c[1] - self.y))
            self.plan = self.plan_route(target)

        if self.plan:
            return self.plan.pop(0)
        return random.choice(["up", "down", "left", "right"])

    def act(self, action: str):
        """Execute one action and handle food collection."""
        if action == "up" and self.y > 0:
            self.y -= 1
        elif action == "down" and self.y < self.env.height - 1:
            self.y += 1
        elif action == "left" and self.x > 0:
            self.x -= 1
        elif action == "right" and self.x < self.env.width - 1:
            self.x += 1

        # Collect food if any
        if self.env.has_food(self.x, self.y):
            self.env.collect_food(self.x, self.y)
            self.collected_food += 1
            self.energy += 20
            self.plan = []  # Clear current plan

        self.energy -= 1

    def update(self):
        """Sense → Decide → Act."""
        if self.energy <= 0:
            return
        perception = self.perceive()
        action = self.decide(perception)
        self.act(action)


class GatheringEnvironment:
    """Grid environment with food and obstacles."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.food = {}       # {(x, y): value}
        self.obstacles = set()

        # Random food
        for _ in range(10):
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            self.food[(x, y)] = random.randint(1, 3)

        # Random obstacles
        for _ in range(8):
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            self.obstacles.add((x, y))

    def is_valid(self, x, y) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def has_obstacle(self, x, y) -> bool:
        return (x, y) in self.obstacles

    def has_food(self, x, y) -> bool:
        return (x, y) in self.food

    def collect_food(self, x, y):
        if (x, y) in self.food:
            del self.food[(x, y)]

    def get_visible_food(self, x, y, radius: int):
        visible = []
        for (fx, fy) in self.food:
            dist = abs(fx - x) + abs(fy - y)
            if dist <= radius:
                visible.append((fx, fy))
        return visible

    def show(self, agent: CollectorAgent):
        for yy in range(self.height):
            row = []
            for xx in range(self.width):
                if xx == agent.x and yy == agent.y:
                    row.append("🤖")
                elif (xx, yy) in self.obstacles:
                    row.append("🧱")
                elif (xx, yy) in self.food:
                    row.append("🍎")
                else:
                    row.append("⬜")
            print(" ".join(row))
        print()


def simulate_gathering(steps: int = 30):
    env = GatheringEnvironment(8, 8)
    agent = CollectorAgent(0, 0, env)

    print("=== SIMULATION: GOAL-BASED AGENT ===\n")
    print("Initial state:")
    env.show(agent)

    for step in range(steps):
        agent.update()

        if step % 5 == 0:
            print(f"\nStep {step + 1}:")
            env.show(agent)
            print(f"Food: {agent.collected_food} | Energy: {agent.energy}")

        if agent.energy <= 0:
            print("\nAgent ran out of energy.")
            break
        if len(env.food) == 0:
            print("\nAll food has been collected!")
            break

    print("\nFinal result:")
    print(f"Collected food: {agent.collected_food}")
    print(f"Remaining energy: {agent.energy}")


if __name__ == "__main__":
    simulate_gathering()
