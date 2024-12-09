import os
import pddlpy
import tempfile
from queue import PriorityQueue



def validate_plan(plan, dp):
    """
    Validates a plan for the given PDDL domain and problem.
    
    Parameters:
    - plan: A list of actions (strings) representing the plan.
    - dp: A DomainProblem object representing the parsed PDDL domain and problem.

    Returns:
    - True if the plan is valid and achieves the goal, raises an error otherwise.
    """
    state = dp.initialstate()  # Get the initial state as a set of atoms
    
    for action_str in plan:
        # Parse the action and arguments
        action_parts = action_str.strip("()").split()
        action_name = action_parts[0]
        action_args = action_parts[1:]

        # Get all grounded operators for the action
        grounded_operators = dp.ground_operator(action_name)
        
        # Find the matching grounded operator
        matching_operator = None
        for op in grounded_operators:
            if list(op.variable_list.values()) == action_args:
                matching_operator = op
                break
        
        if not matching_operator:
            raise ValueError(f"Action {action_str} is not valid or improperly grounded.")
        
        # Check if preconditions are satisfied
        if not ({repr(p) for p in matching_operator.precondition_pos} <= {repr(p) for p in state} and
                {repr(p) for p in matching_operator.precondition_neg}.isdisjoint({repr(p) for p in state})):
            raise ValueError(f"Preconditions not satisfied for action: {action_str}")
        
        # Apply the effects of the action
        state -= matching_operator.effect_neg  # Remove negative effects
        state |= matching_operator.effect_pos  # Add positive effects
    
    # Check if the goal is satisfied in the final state
    goals = dp.goals()
    if not ({repr(p) for p in goals} <= {repr(p) for p in state}):
        return False
    
    return True


def eval_solution(problem: str, domain: str, solution: str):
    # construct a temporary domain and problem file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(domain)
        domain_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(problem)
        problem_file = f.name

    # evaluate the solution
    try:
        dp = pddlpy.DomainProblem(domain_file, problem_file)
        is_valid = validate_plan(solution, dp)
    except Exception as e:
        os.remove(domain_file)
        os.remove(problem_file)
        return False

    os.remove(domain_file)
    os.remove(problem_file)

    return is_valid


def eval_solution_files(problem_file: str, domain_file: str, solution: str):
    # evaluate the solution
    try:
        dp = pddlpy.DomainProblem(domain_file, problem_file)
        is_valid = validate_plan(solution, dp)
    except Exception as e:
        return False

    return is_valid


def heuristic(state, goals):
    """
    A simple heuristic function that returns the number of unsatisfied goals.
    """
    return len(goals - state)

def plan_search(dp):
    """
    Finds a valid plan for the given PDDL domain and problem using A* search.
    
    Parameters:
    - dp: A DomainProblem object representing the parsed PDDL domain and problem.
    
    Returns:
    - A list of actions (strings) that form a valid plan, or None if no plan is found.
    """
    # Get initial state and goals
    initial_state = {repr(p) for p in dp.initialstate()}
    goals = {repr(p) for p in dp.goals()}
    
    # Priority queue for A* search
    open_set = PriorityQueue()
    open_set.put((0, initial_state, []))  # (priority, state, plan)
    
    # Set to keep track of visited states
    visited = set()
    
    while not open_set.empty():
        _, current_state, plan = open_set.get()
        
        # Check if goals are satisfied
        if goals <= current_state:
            return plan
        
        # Mark the state as visited
        state_key = frozenset(current_state)
        if state_key in visited:
            continue
        visited.add(state_key)
        
        # Expand the state by applying all grounded actions
        for action_name in dp.operators():
            grounded_actions = dp.ground_operator(action_name)
            for action in grounded_actions:
                preconditions_pos = {repr(p) for p in action.precondition_pos}
                preconditions_neg = {repr(p) for p in action.precondition_neg}
                
                # Check if preconditions are satisfied
                if not (preconditions_pos <= current_state and preconditions_neg.isdisjoint(current_state)):
                    continue
                
                # Compute the new state after applying the action
                # print("satisfied", action, current_state)
                new_state = set(current_state)
                new_state -= {repr(p) for p in action.effect_neg}
                new_state |= {repr(p) for p in action.effect_pos}
                
                # Create the new plan by appending the current action
                new_plan = plan + [f"({action_name} {' '.join(action.variable_list.values())})"]
                
                # Add the new state to the priority queue
                priority = len(new_plan) + heuristic(new_state, goals)
                open_set.put((priority, new_state, new_plan))
    
    # Return None if no plan is found
    return None


def find_solution(problem: str, domain: str):
    # construct a temporary domain and problem file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="ascii") as f:
        f.write(domain)
        domain_file = f.name
        f.seek(0)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="ascii") as f:
        f.write(problem)
        problem_file = f.name
        f.seek(0)

    # find the solution
    try:
        dp = pddlpy.DomainProblem(domain_file, problem_file)
        plan = plan_search(dp)
    except Exception as e:
        print(e)
        os.remove(domain_file)
        os.remove(problem_file)
        return None

    os.remove(domain_file)
    os.remove(problem_file)

    return plan
