class OutputsBag:
    outputs: dict[str, str]

    def __init__(self, outputs: dict[str, str]):
        self.outputs = outputs
    
    def get_output(self, name: str) -> str:
        return self.outputs.get(name, "")
    
    def set_output(self, name: str, value: str):
        self.outputs[name] = value

    def to_dict(self):
        return self.outputs


class Action:
    name: str
    instructions: str
    outputs: list[str]
    uttered: bool = False
    utterance: str
    def __init__(self, name: str, instructions: str, utterance: str, outputs: list[str]):
        self.name = name
        self.instructions = instructions
        self.uttered = False
        self.utterance = utterance
        self.outputs = outputs

    def __str__(self):
        return f"Action(name={self.name}, instructions={self.instructions}, utterance={self.utterance}, outputs={self.outputs})"
    
    def to_dict(self):
        """Convert the Action object to a dictionary for JSON serialization"""
        return {
            "name": self.name,
            "instructions": self.instructions,
            "uttered": self.uttered,
            "utterance": self.utterance,
            "outputs": self.outputs
        }

class StateTracker:
    actions: list[Action]

    def __init__(self, actions: list[Action], pushback_actions: list[Action]):
        self.actions = actions
        self.pushback_actions = pushback_actions
        self.current_action_index = 0

    def get_current_action(self) -> Action:
        return self.actions[self.current_action_index]
    
    def get_pushback_action(self) -> list[Action]:
        return self.pushback_actions

    def next_action(self):
        self.current_action_index += 1