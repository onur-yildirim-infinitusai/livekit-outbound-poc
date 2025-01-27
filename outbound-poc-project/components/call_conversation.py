from typing import Optional, Callable, Any
from dataclasses import dataclass

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



ANSWER_ACTIONS = [
    Action(name="SayCompanyIndustry", instructions="Say the company industry", utterance="The company industry is health insurance.", outputs=[]),
    Action(name="SayCompanyName", instructions="Say the company name", utterance="The company name is Zing Health", outputs=[]),
    Action(name="SayIDontKnow", instructions="Say that you don't know the answer to the question", utterance="I don't know the answer to that question.", outputs=[]),
    Action(name="IntroduceYourself", instructions="Introduce yourself to the user", utterance="I'm Eva, a digital assistant from Zing Health.", outputs=["is_user_free_to_talk"]),
]

PUSHBACK_ACTIONS = [
    Action(name="AskUserToRepeat", instructions="Pushback the user", utterance="I'm sorry, I didn't catch that. Can you repeat it?", outputs=[]),
    Action(name="ExplainTheReasonOfTheCall", instructions="Explain why we are asking these questions", utterance="We need these questions to give you the best service possible.", outputs=[]),
    Action(name="WillCallback", instructions="Tell the user that you will call them back", utterance="I will call you back in a few minutes.", outputs=[]),
    Action(name="WaitForUserResponse", instructions="Wait for the user response", utterance="Yeah, I can wait.", outputs=[]),
]

def get_moving_forward_actions() -> dict[str, Action]:
    return {
        "SayGreeting": Action(name="SayGreeting", instructions="Greet the user!", utterance="Hello! I want to ask you a few questions! Are you free to talk?", outputs=["is_user_free_to_talk"]),
        "AskForDateOfBirth": Action(name="AskForDateOfBirth", instructions="Ask the user for their date of birth!", utterance="What is your date of birth?", outputs=["date_of_birth"]),
        "AskForFirstName": Action(name="AskForFirstName", instructions="Ask the user for their first name!", utterance="What is your first name?", outputs=["first_name"]),
        "AskForLastName": Action(name="AskForLastName", instructions="Ask the user for their last name!", utterance="What is your last name?", outputs=["last_name"]),
        "EndCall": Action(name="EndCall", instructions="Say goodbye to the user and hang up the call!", utterance="Thank you for your responses. Goodbye!", outputs=[]),
        "ScheduleCallback": Action(name="ScheduleCallback", instructions="Schedule a callback with the user", utterance="I'll call you back at a better time. Goodbye!", outputs=[]),
    }

OUTPUTS_MAP = {
    "date_of_birth": "",
    "first_name": "",
    "last_name": "",
    "is_user_free_to_talk": "",
} 
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

# Action List
class StateTracker:
    actions: list[Action]

    def __init__(self, actions: list[Action], pushback_actions: list[Action], answer_actions: list[Action], outputs_bag: OutputsBag):
        self.actions = actions
        self.pushback_actions = pushback_actions
        self.answer_actions = answer_actions
        self.outputs_bag = outputs_bag
        self.current_action_index = 0

    def get_current_action(self) -> Action:
        return self.actions[self.current_action_index]
    
    def get_pushback_actions(self) -> list[Action]:
        return self.pushback_actions
    
    def get_answer_actions(self) -> list[Action]:
        return self.answer_actions
    
    def get_outputs_bag(self) -> OutputsBag:
        return self.outputs_bag

    def next_action(self):
        self.current_action_index += 1

# Action Tree with Nodes
@dataclass(frozen=True)
class Condition:
    """Represents a condition to evaluate for determining the next action"""
    check: Callable[[OutputsBag], bool]
    description: str

    def __hash__(self) -> int:
        return hash(self.description)  # Using description as the unique identifier
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Condition):
            return NotImplemented
        return self.description == other.description
    
class ActionNode:
    def __init__(self, action: Action, conditions: dict[Condition, 'ActionNode'] | None = None, default_next: Optional['ActionNode'] = None):
        self.action = action
        self.conditions = conditions or {}  # Map of Condition -> next ActionNode
        self.default_next = default_next

class ActionTree:
    def __init__(self, root: ActionNode):
        self.root = root
        self.current_node = root

    def get_current_action(self) -> Action:
        return self.current_node.action

    def next_action(self, outputs_bag: OutputsBag) -> Optional[Action]:
        """Determine and move to the next action based on conditions"""
        if not self.current_node.conditions and not self.current_node.default_next:
            return None  # End of tree reached

        # Check all conditions
        for condition, next_node in self.current_node.conditions.items():
            if condition.check(outputs_bag):
                self.current_node = next_node
                return next_node.action

        # If no conditions match, use default path
        if self.current_node.default_next:
            self.current_node = self.current_node.default_next
            return self.current_node.action

        return None

class StateTrackerForActionTree:
    def __init__(self, action_tree: ActionTree, pushback_actions: list[Action], answer_actions: list[Action], outputs_bag: OutputsBag):
        self.action_tree = action_tree
        self.pushback_actions = pushback_actions
        self.answer_actions = answer_actions
        self.outputs_bag = outputs_bag

    def get_current_action(self) -> Action:
        return self.action_tree.get_current_action()

    def get_pushback_actions(self) -> list[Action]:
        return self.pushback_actions
    
    def get_answer_actions(self) -> list[Action]:
        return self.answer_actions
    
    def get_outputs_bag(self) -> OutputsBag:
        return self.outputs_bag

    def next_action(self) -> Optional[Action]:
        return self.action_tree.next_action(self.outputs_bag)

# Initialize Conversation with Action List & Call Outputs Bag
def initialize_conversation() -> StateTracker:
    outputs_bag = OutputsBag(OUTPUTS_MAP)

    moving_forward_actions = get_moving_forward_actions()

    actions = [
        moving_forward_actions["SayGreeting"],
        moving_forward_actions["AskForDateOfBirth"],
        moving_forward_actions["AskForFirstName"],
        moving_forward_actions["AskForLastName"],
        moving_forward_actions["EndCall"],
    ]

    state_tracker = StateTracker(actions, PUSHBACK_ACTIONS, ANSWER_ACTIONS, outputs_bag)

    return state_tracker

# Initialize Conversation with Action Tree & Outputs Bag

def create_conversation_tree() -> ActionTree:
    # conditions
    is_free_to_talk = Condition(
        lambda bag: bag.get_output("is_user_free_to_talk") == "yes",
        "User is free to talk"
    )
    
    is_not_free_to_talk = Condition(
        lambda bag: bag.get_output("is_user_free_to_talk") == "no",
        "User is not free to talk"
    )

    is_dob_provided = Condition(
        lambda bag: bag.get_output("date_of_birth") != "",
        "Date of birth is provided"
    )

    is_first_name_provided = Condition(
        lambda bag: bag.get_output("first_name") != "",
        "First name is provided"
    )

    is_last_name_provided = Condition(
        lambda bag: bag.get_output("last_name") != "",
        "Last name is provided"
    )

    # actions
    moving_forward_actions = get_moving_forward_actions()
    end_call_action = moving_forward_actions["EndCall"]
    callback_action = moving_forward_actions["ScheduleCallback"]

    # link actions and conditions in action nodes
    end_node = ActionNode(end_call_action)
    callback_node = ActionNode(callback_action)

    dob_node = ActionNode(
        moving_forward_actions["AskForDateOfBirth"],
        conditions={
            is_dob_provided: end_node,
        }
    )
    
    greeting_node = ActionNode(
        moving_forward_actions["SayGreeting"],
        conditions={
            is_free_to_talk: dob_node,
            is_not_free_to_talk: callback_node
        }
    )

    return ActionTree(greeting_node)
    
def initialize_conversation_with_action_tree() -> StateTrackerForActionTree:
    outputs_bag = OutputsBag(OUTPUTS_MAP)
    action_tree = create_conversation_tree()
    state_tracker = StateTrackerForActionTree(action_tree, PUSHBACK_ACTIONS, ANSWER_ACTIONS, outputs_bag)
    return state_tracker