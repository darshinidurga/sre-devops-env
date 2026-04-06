import sys; sys.path.insert(0,'.'); sys.path.insert(0,'./server')
from simulator import SRESimulator
from models import Action, ActionType
import pydantic

sim = SRESimulator()
obs = sim.reset('easy')
act = Action(action_type=ActionType.RestartService, target_id='web-3')

try:
    resp = sim.step(act)
    print('OK', resp)
except pydantic.ValidationError as e:
    print('Validation Error Details:')
    for err in e.errors():
        print(f'field: {err.get("loc")}, msg: {err.get("msg")}, type: {err.get("type")}')
