import pulp
import pandas as pd
import numpy as np
from argparse import ArgumentParser

NUM_STUDENTS = 15
NUM_EVENTS = 23
MAX_EVENTS_PER_STUDENT = 4
MIN_EVENTS_PER_STUDENT = 3

def parse_args() -> tuple[str, str, str]:
    parser = ArgumentParser(description='Optimize schedule')
    parser.add_argument("-p", "--preferences", default="./preferences.csv", help="CSV containing the preferences of students for different events")
    parser.add_argument("-s", "--schedule", default="./schedule.csv", help="CSV containing the schedule of events (events per slot)")
    parser.add_argument("-e", "--events", default="./events.csv", help="CSV containing the number of competitors allowed in each event")
    parser.add_argument("-o", "--output", default="./output.csv", help="Path to write the output CSV to")
    args = parser.parse_args()
    return args.preferences, args.schedule, args.events, args.output

def read_preferences(preferences_file: str) -> list[int]:
    return pd.read_csv(preferences_file, index_col=0).fillna(0).astype(int).melt().value.to_list()

def read_schedule(schedule_file: str) -> list[list[int]]:
    row_numbers = []
    schedule = pd.read_csv(schedule_file,index_col=0).fillna(0).astype(int)
    for col in schedule.columns:
        row_numbers.append(schedule.loc[schedule[col] == 1].index.values.tolist())
    return row_numbers

def read_events(events_file: str) -> list[int]:
    return pd.read_csv(events_file,index_col=0).iloc[:, 0].tolist()

def solve(preferences: list[int], schedule: list[list[int]], events: list[int]) -> list[int]:
    problem = pulp.LpProblem('assign_events', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', range(NUM_STUDENTS * NUM_EVENTS), lowBound=0, upBound=1, cat='Integer')
    problem.setObjective(sum([sum([preferences[NUM_EVENTS * i + j]*x[NUM_EVENTS * i + j] for j in range(NUM_EVENTS)])for i in range(NUM_STUDENTS)]))

    # students can have a range of different numbers of events, defined with these config variables TODO: change to be like events file, more flexible
    for student in range(NUM_STUDENTS):
        problem.addConstraint(sum([x[NUM_EVENTS * student + event] for event in range(NUM_EVENTS)]) <= MAX_EVENTS_PER_STUDENT)
        problem.addConstraint(sum([x[NUM_EVENTS * student + event] for event in range(NUM_EVENTS)]) >= MIN_EVENTS_PER_STUDENT)

    # events have 2 or 3 people, as defined in events file
    for event in range(NUM_EVENTS):
        problem.addConstraint(sum([x[NUM_EVENTS * student + event] for student in range(NUM_STUDENTS)]) == events[event])

    # students cannot have positive assignments to events in the same slot (i.e., student cannot be assigned to multiple events in the same slot)
    for slot in schedule:
        for student in range(NUM_STUDENTS):
            problem.addConstraint(sum([x[NUM_EVENTS * student + event] for event in slot]) <= 1)
    
    # solve problem and return found values
    flag = problem.solve()
    if flag == -1:
        print("Could not find optimal solution")
        raise Exception()
    
    return [x[i].value() for i in range(NUM_EVENTS * NUM_STUDENTS)]


def write_output(assignments: list[int], output_file: str) -> None:
    pd.DataFrame(np.array(assignments).reshape(15, 23)).astype(int).to_csv(output_file)

if __name__ == '__main__':

    # Read in the preferences, schedule, and events data
    preferences_file, schedule_file, events_file, output_file = parse_args()
    preferences = read_preferences(preferences_file)
    schedule = read_schedule(schedule_file)
    events = read_events(events_file)

    # Solve and write the output to desired file
    assignments = solve(preferences, schedule, events)
    write_output(assignments, output_file)


