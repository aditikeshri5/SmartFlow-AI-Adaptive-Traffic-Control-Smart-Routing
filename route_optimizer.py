
import heapq
import math


class RouteOptimizer:
    """
    A* route optimizer for a sample road network.

    Shortest mode minimizes distance.
    Fastest mode minimizes estimated travel time, including
    congestion and signal delays.
    """

    def __init__(self):
        # Every road is listed in both directions with matching values.
        self.graph = {
            "A": {
                "B": {
                    "distance_km": 4,
                    "speed_kmph": 40,
                    "congestion": 80,
                    "signal_delay": 50,
                },
                "C": {
                    "distance_km": 3,
                    "speed_kmph": 40,
                    "congestion": 10,
                    "signal_delay": 5,
                },
            },
            "B": {
                "A": {
                    "distance_km": 4,
                    "speed_kmph": 40,
                    "congestion": 80,
                    "signal_delay": 50,
                },
                "D": {
                    "distance_km": 3,
                    "speed_kmph": 35,
                    "congestion": 70,
                    "signal_delay": 40,
                },
                "E": {
                    "distance_km": 5,
                    "speed_kmph": 45,
                    "congestion": 20,
                    "signal_delay": 10,
                },
            },
            "C": {
                "A": {
                    "distance_km": 3,
                    "speed_kmph": 40,
                    "congestion": 10,
                    "signal_delay": 5,
                },
                "D": {
                    "distance_km": 5,
                    "speed_kmph": 45,
                    "congestion": 10,
                    "signal_delay": 5,
                },
            },
            "D": {
                "B": {
                    "distance_km": 3,
                    "speed_kmph": 35,
                    "congestion": 70,
                    "signal_delay": 40,
                },
                "C": {
                    "distance_km": 5,
                    "speed_kmph": 45,
                    "congestion": 10,
                    "signal_delay": 5,
                },
                "E": {
                    "distance_km": 2,
                    "speed_kmph": 40,
                    "congestion": 15,
                    "signal_delay": 5,
                },
            },
            "E": {
                "B": {
                    "distance_km": 5,
                    "speed_kmph": 45,
                    "congestion": 20,
                    "signal_delay": 10,
                },
                "D": {
                    "distance_km": 2,
                    "speed_kmph": 40,
                    "congestion": 15,
                    "signal_delay": 5,
                },
            },
        }

    @staticmethod
    def road_cost(road):
        """Calculate estimated travel cost in seconds."""

        driving_time = (
            road["distance_km"] / road["speed_kmph"]
        ) * 3600

        congestion_delay = (
            driving_time * road["congestion"] / 100 * 0.8
        )

        return (
            driving_time
            + congestion_delay
            + road["signal_delay"]
        )

    def heuristic(self, node, goal, mode):
        """
        A safe heuristic for this sample graph.

        Zero is admissible for both distance and time.
        A* therefore behaves like Dijkstra's algorithm,
        guaranteeing an optimal route for nonnegative costs.
        """
        return 0

    def find_route(self, start, goal, mode="fastest"):
        if start not in self.graph or goal not in self.graph:
            raise ValueError(
                "Start and goal must be one of: A, B, C, D, E."
            )

        if mode not in ("fastest", "shortest"):
            raise ValueError("Mode must be 'fastest' or 'shortest'.")

        if start == goal:
            return {
                "route": [start],
                "distance_km": 0,
                "estimated_time_seconds": 0,
                "explored_nodes": [start],
            }

        # Heap entries: (f_score, g_score, node)
        frontier = [(0, 0, start)]
        came_from = {}
        g_score = {node: math.inf for node in self.graph}
        g_score[start] = 0
        explored = []

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)

            # Ignore outdated heap entries.
            if current_cost > g_score[current]:
                continue

            explored.append(current)

            if current == goal:
                break

            for neighbour, road in self.graph[current].items():

                if mode == "shortest":
                    step_cost = road["distance_km"]
                else:
                    step_cost = self.road_cost(road)

                tentative_g = current_cost + step_cost

                if tentative_g < g_score[neighbour]:
                    g_score[neighbour] = tentative_g
                    came_from[neighbour] = current

                    f_score = tentative_g + self.heuristic(
                        neighbour, goal, mode
                    )

                    heapq.heappush(
                        frontier,
                        (f_score, tentative_g, neighbour),
                    )

        if goal not in came_from:
            raise ValueError("No route found.")

        # Reconstruct the route.
        route = [goal]

        while route[-1] != start:
            route.append(came_from[route[-1]])

        route.reverse()

        # Calculate distance and travel time for the selected route.
        total_distance = 0
        total_time = 0

        for i in range(len(route) - 1):
            road = self.graph[route[i]][route[i + 1]]
            total_distance += road["distance_km"]
            total_time += self.road_cost(road)

        return {
            "route": route,
            "distance_km": round(total_distance, 2),
            "estimated_time_seconds": round(total_time, 2),
            "explored_nodes": explored,
        }


if __name__ == "__main__":
    optimizer = RouteOptimizer()

    for mode in ("shortest", "fastest"):
        result = optimizer.find_route("A", "E", mode=mode)

        print(f"\n--- {mode.upper()} ROUTE ---")
        print("Path:", " -> ".join(result["route"]))
        print("Distance:", result["distance_km"], "km")
        print(
            "Estimated travel time:",
            round(result["estimated_time_seconds"] / 60, 2),
            "minutes",
        )
        print("Explored nodes:", result["explored_nodes"])
