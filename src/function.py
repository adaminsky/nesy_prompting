from .symbol_mapping import prompting_mapper, prompting_mapper_structure


class Function:
    def __init__(self, model, processor, fn, *args):
        self.model = model
        self.processor = processor
        self.fn = fn
        self.args = args
        self.symbol_mapper = prompting_mapper
        self.symbol_mapper_structure = prompting_mapper_structure
        self.learned_mapper = None

    def __call__(self, *args):
        return self.fn(*args)

    def get_arg_symbols(self, i):
        assert i < len(
            self.args
        ), f"Index {i} is out of bounds for {len(self.args)} arguments"
        return self.args[i]

    def nesy_v1(self, *inputs):
        """Rather than symbolic input, we now have raw (non-symbolic) inputs
        corresponding to each of the original inputs to the function."""

        assert len(inputs) == len(
            self.args
        ), f"Expected {len(self.args)} inputs, got {len(inputs)}"
        symbols = []
        for i in range(len(self.args)):
            symbols.append(
                self.symbol_mapper(
                    inputs[i], self.get_arg_symbols(i), self.model, self.processor
                )
            )

        return self.fn(*symbols)

    def nesy_v2(self, input):
        """Rather than symbolic input, we now have a single non-symbolic input
        containing all the required information."""

        symbols = []
        for i in range(len(self.args)):
            symbols.append(
                self.symbol_mapper_structure(
                    input, i, self.get_arg_symbols(i), self.model, self.processor
                )
            )

        return self.fn(*symbols), *symbols
