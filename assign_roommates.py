import pulp
import pandas as pd
import numpy as np
from argparse import ArgumentParser


def parse_args() -> tuple[str, str, str]:
    parser = ArgumentParser(description='Assign roommate pairs')
    parser.add_argument("-p", "--preferences", default="./preferences.csv", help="CSV containing the preferences of students for different rooommates")
    parser.add_argument("-r", "--rooms", default="./rooms.csv", help="CSV containing the rooms available")
    parser.add_argument("-o", "--output", default="./output.csv", help="Path to write the output CSV to")
    args = parser.parse_args()
    return args.preferences, args.rooms, args.output

def read_rooms(rooms_file: str) -> np.ndarray:
    return pd.read_csv(rooms_file, index_col=0).astype(int).to_numpy().flatten()

def read_preferences(preferences_file: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(preferences_file)
    return df.drop("Name",axis=1).astype(int).to_numpy(), df.loc[:,"Name"].to_numpy()

def calculate_compatibility_array(preferences: np.ndarray, num_residents: int=-1, num_preference_categories: int=-1) -> np.ndarray:
    def calculate_compatibility(a_prefs, b_prefs):
        return np.dot(np.array([5, 4, 3, 2, 1]), np.abs(a_prefs - b_prefs))

    compatibilities = []

    for a, a_prefs in enumerate(preferences):
        a_compatibilities = []
        for b, b_prefs in enumerate(preferences):
            if a == b:
                a_compatibilities.append(1000000)
            else:
                a_compatibilities.append(calculate_compatibility(a_prefs, b_prefs))
        compatibilities.append(a_compatibilities)
    
    return np.array(compatibilities)

def solve(preferences: np.ndarray, rooms: np.ndarray) -> np.ndarray:
    problem = pulp.LpProblem('lpsolver', pulp.LpMinimize)
    x  = pulp.LpVariable.dicts('x', range(NUM_RESIDENTS * NUM_RESIDENTS * NUM_ROOMS), lowBound=0, upBound=1, cat='Integer')
    compatibility_array = calculate_compatibility_array(preferences)
    problem.setObjective(
        sum([
            sum([
                sum([
                    compatibility_array[s][t]*x[s * (NUM_ROOMS * NUM_RESIDENTS) +  t * (NUM_ROOMS) + j]
                for t in range(NUM_RESIDENTS)]) 
            for s in range(NUM_RESIDENTS)]) 
        for j in range(NUM_ROOMS)]))

    for s in range(NUM_RESIDENTS):
        sum_expr = sum([sum([x[s * (NUM_ROOMS * NUM_RESIDENTS) +  t * (NUM_ROOMS) + j] for j in range(NUM_ROOMS)]) for t in range(NUM_RESIDENTS)])
        problem.addConstraint(sum_expr == 1)

    for j in range(NUM_ROOMS):
        sum_expr = sum([sum([x[s * (NUM_ROOMS * NUM_RESIDENTS) +  t * (NUM_ROOMS) + j] for s in range(NUM_RESIDENTS)]) for t in range(NUM_RESIDENTS)])
        problem.addConstraint(sum_expr == 2)

    for j in range(NUM_ROOMS):
        for s in range(NUM_RESIDENTS):
            for t in range(NUM_RESIDENTS):
                problem.addConstraint(x[s * (NUM_ROOMS * NUM_RESIDENTS) +  t * (NUM_ROOMS) + j] == x[t * (NUM_ROOMS * NUM_RESIDENTS) + s * (NUM_ROOMS) + j])
    
    flag = problem.solve()
    if flag == -1:
        print("Could not find optimal solution")
        raise Exception()
    
    return np.array([x[i].value() for i in range(len(x))])


def write_output(assignments: list[int], names, output_file: str) -> None:
    reshaped = assignments.reshape((NUM_RESIDENTS, NUM_RESIDENTS, NUM_ROOMS))

    matchings = []

    for s in range(reshaped.shape[0]):
        for t in range(reshaped[s].shape[0]):
            if np.any(reshaped[s][t]):
                if s < t:
                    j = np.where(reshaped[s][t] == 1.0)[0][0]
                    matchings.append((names[s], names[t], j))

    pd.DataFrame(matchings).to_csv("./output.csv",index=False, header=["Roommate A", "Roommate B", "Room"])


if __name__ == '__main__':

    # Read in the preferences, names, and rooms data
    preferences_file, rooms_file, output_file = parse_args()
    preferences, names = read_preferences(preferences_file)
    rooms = read_rooms(rooms_file)

    NUM_RESIDENTS = preferences.shape[0]
    NUM_ROOMS = rooms.shape[0]

    # Solve and write the output to desired file
    assignments = solve(preferences, rooms)
    write_output(assignments, names, output_file)



