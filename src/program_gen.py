import random
from dataclasses import dataclass
from typing import List, Any, Tuple, Set, Optional
import itertools

random.seed(0)


@dataclass
class DSLOp:
    name: str
    arg_indices: List[int]  # Indices of list elements to use as arguments
    output_type: str
    implementation: callable


class ProgramSynthesisProblem:
    def __init__(
        self,
        inputs: List[List[int]],
        expected_outputs: List[Any],
        available_ops: List[DSLOp],
        input_size: int,
    ):
        self.inputs = inputs
        self.expected_outputs = expected_outputs
        self.available_ops = available_ops
        self.input_size = input_size


class ProblemGenerator:
    def __init__(self):
        # Basic DSL operations that work with list indices
        self.base_ops = [
            DSLOp("add", [0, 1], "int", lambda x, y: x + y),
            DSLOp("subtract", [0, 1], "int", lambda x, y: x - y),
            DSLOp("multiply", [0, 1], "int", lambda x, y: x * y),
            # DSLOp("divide", [0, 1], "int", lambda x, y: x // y if y != 0 else 0),
            # DSLOp("mod", [0, 1], "int", lambda x, y: x % y if y != 0 else 0),
            DSLOp("negate", [0], "int", lambda x: -x),
            # DSLOp("is_positive", [0], "bool", lambda x: x > 0),
            DSLOp("max", [0, 1], "int", max),
            DSLOp("min", [0, 1], "int", min),
            # DSLOp("and", [0, 1], "bool", lambda x, y: x and y),
            # DSLOp("or", [0, 1], "bool", lambda x, y: x or y),
            # DSLOp("not", [0], "bool", lambda x: not x)
        ]

    def generate_random_input(self, size: int = 5) -> List[int]:
        """Generate a random input list"""
        return [random.randint(-10, 10) for _ in range(size)]

    def compose_operations(
        self,
        depth: int,
        available_ops: List[DSLOp],
        used_indices: Set[int],
        input_size: int,
        input_type: str = "int",
    ) -> Tuple[callable, str, Set[int]]:
        """Generate a random composition of operations with specified depth"""
        if depth <= 1 or len(used_indices) >= input_size:
            # Use a new input index if available
            remaining_indices = set(range(input_size)) - used_indices
            if remaining_indices:
                idx = random.choice(list(remaining_indices))
                used_indices.add(idx)
                return (lambda lst, i=idx: lst[i]), f"input[{idx}]", used_indices
            else:
                idx = random.choice(list(used_indices))
                return (lambda lst, i=idx: lst[i]), f"input[{idx}]", used_indices

        # Randomly choose an operation that outputs the desired type
        valid_ops = [op for op in available_ops if op.output_type == input_type]
        if not valid_ops:
            raise ValueError(f"No operations available for type {input_type}")

        op = random.choice(valid_ops)
        sub_funcs = []
        sub_names = []
        current_indices = used_indices.copy()

        # Recursively generate subfunctions
        for _ in range(len(op.arg_indices)):
            sub_func, sub_name, current_indices = self.compose_operations(
                depth - 1, available_ops, current_indices, input_size, input_type
            )
            sub_funcs.append(sub_func)
            sub_names.append(sub_name)

        # Create the composed function
        def composed_func(lst):
            intermediate_results = [f(lst) for f in sub_funcs]
            return op.implementation(*intermediate_results)

        composed_name = f"{op.name}({', '.join(sub_names)})"
        return composed_func, composed_name, current_indices

    def generate_problem(
        self, depth: int = 3, width: int = 5, num_examples: int = 6, input_size: int = 5
    ) -> Tuple[ProgramSynthesisProblem, str]:
        """Generate a program synthesis problem with specified complexity"""
        if width > len(self.base_ops):
            width = len(self.base_ops)

        # Randomly select operations for the DSL
        available_ops = random.sample(self.base_ops, width)

        # Generate the target function and its description
        target_func, target_desc, _ = self.compose_operations(
            depth, available_ops, set(), input_size
        )

        # Generate input-output examples
        inputs = [self.generate_random_input(input_size) for _ in range(num_examples)]
        outputs = [target_func(input_vals) for input_vals in inputs]

        return (
            ProgramSynthesisProblem(inputs, outputs, available_ops, input_size),
            target_desc,
        )


def demonstrate_generator(min_width=1, max_width=5, min_depth=1, max_depth=5):
    generator = ProblemGenerator()

    # Generate problems with increasing complexity
    data = []
    for depth in range(1 + min_depth, 2 + max_depth):
        for width in range(min_width, 1 + max_width):
            for _ in range(16):
                while True:
                    try:
                        problem, solution = generator.generate_problem(
                            depth=depth,
                            width=width,
                            input_size=5,  # Each input will be a list of 5 integers
                        )
                        break
                    except:
                        continue

                problem_desc = "Available operations:\n"
                for op in problem.available_ops:
                    problem_desc += f"  - {op.name}: ({', '.join(['int' for _ in range(len(op.arg_indices))])}) -> {op.output_type}\n"
                problem_desc += "Each input to the above operations is a value from the list 'input'. The values are accessed by their index in the list. For example, 'input[0]' is the first value in the list. A function f: int -> int can be called like f(input[4]) which would apply f to the 5th value in input.\n"
                problem_desc += "\nInput-Output Examples:\n"
                for i, (inp, out) in enumerate(
                    zip(problem.inputs[:5], problem.expected_outputs[:5])
                ):
                    problem_desc += f"  Example {i + 1}: {inp} -> {out}\n"
                problem_desc += "Given the above examples, what is the output for the following input?\n"
                problem_desc += f"Input: {problem.inputs[-1]} -> ?"
                problem_answer = problem.expected_outputs[-1]
                examples = (problem.inputs[:5], problem.expected_outputs[:5])
                data.append((problem_desc, problem_answer, examples, solution))
    return data


@dataclass
class Expression:
    op: Optional[DSLOp]  # None means this is a leaf node (input access)
    args: List["Expression"]  # Empty for leaf nodes
    input_idx: Optional[int]  # Only used for leaf nodes

    def evaluate(self, input_list: List[int]) -> int:
        if self.op is None:
            return input_list[self.input_idx]
        arg_values = [arg.evaluate(input_list) for arg in self.args]
        return self.op.implementation(*arg_values)

    def to_string(self) -> str:
        if self.op is None:
            return f"input[{self.input_idx}]"
        arg_strs = [arg.to_string() for arg in self.args]
        return f"{self.op.name}({', '.join(arg_strs)})"


class ProgramSynthesisSolver:
    def __init__(self, operations: List[dict]):
        operations = [DSLOp(**op) for op in operations]
        self.operations = operations

    def create_leaf(self, input_idx: int) -> Expression:
        """Create a leaf node that accesses an input at the given index"""
        return Expression(op=None, args=[], input_idx=input_idx)

    def create_node(self, op: DSLOp, args: List[Expression]) -> Expression:
        """Create an internal node with the given operation and arguments"""
        return Expression(op=op, args=args, input_idx=None)

    def enumerate_expressions(self, depth: int, input_size: int) -> List[Expression]:
        """Enumerate all possible expressions up to the given depth"""
        if depth <= 0:
            return [self.create_leaf(i) for i in range(input_size)]

        expressions = self.enumerate_expressions(depth - 1, input_size)
        result = expressions.copy()  # Include all simpler expressions

        # Add new expressions by combining operations with subexpressions
        for op in self.operations:
            for args in itertools.product(expressions, repeat=len(op.arg_indices)):
                result.append(self.create_node(op, list(args)))

        return result

    def verify_expression(
        self, expr: Expression, examples: List[Tuple[List[int], int]]
    ) -> bool:
        """Check if an expression satisfies all input-output examples"""
        return all(expr.evaluate(inputs) == output for inputs, output in examples)

    def solve(
        self, examples: List[Tuple[List[int], int]], max_depth: int = 6
    ) -> Optional[Expression]:
        """Find an expression that satisfies all examples"""
        input_size = len(examples[0][0])  # Size of first input list

        # Try increasingly complex expressions until solution is found
        for depth in range(max_depth + 1):
            expressions = self.enumerate_expressions(depth, input_size)
            for expr in expressions:
                if self.verify_expression(expr, examples):
                    return expr

        return None  # No solution found within max_depth


if __name__ == "__main__":
    random.seed(0)
    data = demonstrate_generator()
    # print(data[0][0])
    # print(data[0][1])
    # print(data[0][2])
    # print()
    # print(data[-1][0])
    # print(data[-1][1])
    # print(data[-1][2])
    min_op = DSLOp(
        name="min", arg_indices=[0, 1], output_type="int", implementation=min
    )

    # Create solver with just the min operation
    solver = ProgramSynthesisSolver([min_op])

    # Define the examples
    examples = [
        ([6, 5, 2, -1, 5], -1),
        ([1, 8, -4, 6, -6], 1),
        ([-1, -6, -7, 9, -2], -1),
        ([7, 9, -6, -1, -7], -1),
        ([-8, 0, 5, 7, -7], -8),
    ]

    # Find a solution
    solution = solver.solve(examples, max_depth=4)
    print(solution.to_string())
