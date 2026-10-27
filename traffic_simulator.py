
import numpy as np

from fuzzy_controller import FuzzyTrafficController


class TrafficSimulator:
    """Simulate a four-way intersection with adaptive traffic signals."""

    APPROACHES = ["North", "East", "South", "West"]

    def __init__(
        self,
        simulation_time=900,
        seed=42,
        fixed_green=30,
        lane_capacity=20,
        service_interval=2,
    ):
        self.simulation_time = int(simulation_time)
        self.rng = np.random.default_rng(seed)
        self.fixed_green = int(fixed_green)
        self.lane_capacity = int(lane_capacity)
        self.service_interval = int(service_interval)

        self.fuzzy_controller = FuzzyTrafficController()

        # Expected vehicle arrivals per second for each approach.
        # These are configurable simulation assumptions.
        self.arrival_rates = {
            "North": 0.30,
            "East": 0.22,
            "South": 0.35,
            "West": 0.18,
        }

    def run_simulation(self, mode="fuzzy"):
        """
        Run the intersection simulation.

        mode='fixed': fixed-time traffic signals
        mode='fuzzy': fuzzy-controlled green durations
        """

        if mode not in ("fixed", "fuzzy"):
            raise ValueError("mode must be 'fixed' or 'fuzzy'")

        queues = {name: [] for name in self.APPROACHES}
        waiting_times = []
        completed_vehicles = 0
        total_arrivals = 0
        max_queue = 0

        current_phase = 0
        phase_remaining = 0
        service_counter = 0

        green_time_history = []
        queue_history = []

        for second in range(self.simulation_time):

            # 1. Generate random vehicle arrivals.
            for approach in self.APPROACHES:
                arrivals = self.rng.poisson(
                    self.arrival_rates[approach]
                )

                for _ in range(arrivals):
                    queues[approach].append(second)
                    total_arrivals += 1

            # 2. Record the largest queue observed.
            current_total_queue = sum(
                len(queue) for queue in queues.values()
            )
            max_queue = max(max_queue, current_total_queue)

            # 3. Decide the green duration at the start of each phase.
            if phase_remaining <= 0:
                approach = self.APPROACHES[current_phase]
                queue = queues[approach]

                if mode == "fixed":
                    green_duration = self.fixed_green
                else:
                    density = min(
                        len(queue) / self.lane_capacity * 100,
                        100,
                    )

                    if queue:
                        waiting = min(second - queue[0], 120)
                    else:
                        waiting = 0

                    result = self.fuzzy_controller.calculate_green_time(
                        density=density,
                        waiting_time=waiting,
                    )

                    green_duration = max(
                        1, round(result["green_time"])
                    )

                phase_remaining = green_duration
                green_time_history.append({
                    "time": second,
                    "approach": approach,
                    "green_seconds": green_duration,
                })

                service_counter = 0

            # 4. Serve vehicles on the currently green approach.
            approach = self.APPROACHES[current_phase]
            queue = queues[approach]

            if queue:
                service_counter += 1

                if service_counter >= self.service_interval:
                    arrival_time = queue.pop(0)
                    waiting_times.append(second - arrival_time)
                    completed_vehicles += 1
                    service_counter = 0

            # 5. Store the queue size for the dashboard.
            queue_history.append({
                "time": second,
                "North": len(queues["North"]),
                "East": len(queues["East"]),
                "South": len(queues["South"]),
                "West": len(queues["West"]),
            })

            # 6. Move to the next approach when green expires.
            phase_remaining -= 1

            if phase_remaining <= 0:
                current_phase = (
                    current_phase + 1
                ) % len(self.APPROACHES)

        remaining_vehicles = sum(
            len(queue) for queue in queues.values()
        )

        average_wait = (
            float(np.mean(waiting_times))
            if waiting_times else 0.0
        )

        return {
            "mode": mode,
            "simulation_time": self.simulation_time,
            "total_arrivals": total_arrivals,
            "vehicles_served": completed_vehicles,
            "remaining_vehicles": remaining_vehicles,
            "average_wait": round(average_wait, 2),
            "max_queue": max_queue,
            "throughput_per_minute": round(
                completed_vehicles
                / (self.simulation_time / 60),
                2,
            ),
            "green_time_history": green_time_history,
            "queue_history": queue_history,
        }


if __name__ == "__main__":
    simulator = TrafficSimulator(simulation_time=900, seed=42)

    fixed_results = simulator.run_simulation(mode="fixed")

    # Use the same random seed for a fair comparison.
    fuzzy_simulator = TrafficSimulator(
        simulation_time=900,
        seed=42,
    )
    fuzzy_results = fuzzy_simulator.run_simulation(mode="fuzzy")

    print("\n--- FIXED-TIME SIGNALS ---")
    print(fixed_results)

    print("\n--- FUZZY-CONTROLLED SIGNALS ---")
    print(fuzzy_results)
