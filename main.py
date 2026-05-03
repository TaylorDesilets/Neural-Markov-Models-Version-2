

import simulated_data
import real_data


def main():
    print("\n--- SIMULATION ---")
    simulated_data.run_simulation()

    print("\n--- REAL DATA ---")
    real_data.run_real_data()

if __name__ == "__main__":
    main()
    