import os
import pddlpy
import tempfile



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