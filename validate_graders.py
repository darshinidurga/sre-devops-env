import sys
sys.path.insert(0, ".")

from server.graders import TaskGrader, run_grader
from server.tasks.easy   import setup_scenario as e_setup
from server.tasks.medium import setup_scenario as m_setup
from server.tasks.hard   import setup_scenario as h_setup
from models import Action, ActionType

grader = TaskGrader()

# 1. Action objects
s = grader.grade_episode("easy", [Action(action_type=ActionType.RestartService, target_id="web-3")], e_setup(), 1)
assert s == 1.0, s
print("PASS 1: Action objects ->", s)

# 2. Plain dicts (HTTP / inference.py style)
s = grader.grade_episode("easy", [{"action_type": "RestartService", "target_id": "web-3"}], e_setup(), 1)
assert s == 1.0, s
print("PASS 2: dict actions ->", s)

# 3. Mixed list
s = grader.grade_episode("easy", [
    {"action_type": "InvestigateLog", "target_id": "web-3"},
    Action(action_type=ActionType.RestartService, target_id="web-3"),
], e_setup(), 2)
assert s == 0.7, s
print("PASS 3: mixed list ->", s)

# 4. Medium dicts
s = grader.grade_episode("medium", [
    {"action_type": "ScaleUp", "target_id": "api-gw-1"},
    {"action_type": "ScaleUp", "target_id": "web-1"},
], m_setup(), 10)
assert s == 1.0, s
print("PASS 4: medium dict actions ->", s)

# 5. Hard dicts
s = grader.grade_episode("hard", [
    {"action_type": "InvestigateLog",     "target_id": "web-1"},
    {"action_type": "InvestigateLog",     "target_id": "web-2"},
    {"action_type": "RollbackDeployment", "target_id": "v2.3.0"},
    {"action_type": "RestartService",     "target_id": "web-1"},
    {"action_type": "RestartService",     "target_id": "web-2"},
], h_setup(), 5)
assert s == 1.0, s
print("PASS 5: hard dict actions ->", s)

# 6. run_grader standalone
s = run_grader("easy", [{"action_type": "RestartService", "target_id": "web-3"}], e_setup(), 1)
assert s == 1.0, s
print("PASS 6: run_grader standalone ->", s)

# 7. Zero score (empty history)
s = grader.grade_episode("easy", [], e_setup(), 0)
assert s == 0.0, s
print("PASS 7: empty history ->", s)

# 8. get_all_scores
scores = grader.get_all_scores({
    "easy":   {"action_history": [{"action_type": "RestartService", "target_id": "web-3"}], "final_state": e_setup(), "ticks_used": 1},
    "medium": {"action_history": [{"action_type": "ScaleUp", "target_id": "api-gw-1"}, {"action_type": "ScaleUp", "target_id": "web-1"}], "final_state": m_setup(), "ticks_used": 10},
    "hard":   {"action_history": [{"action_type": "InvestigateLog", "target_id": "web-1"}, {"action_type": "InvestigateLog", "target_id": "web-2"}, {"action_type": "RollbackDeployment", "target_id": "v2.3.0"}, {"action_type": "RestartService", "target_id": "web-1"}, {"action_type": "RestartService", "target_id": "web-2"}], "final_state": h_setup(), "ticks_used": 5},
})
assert scores == {"easy": 1.0, "medium": 1.0, "hard": 1.0}, scores
print("PASS 8: get_all_scores ->", scores)

# 9. validate_score clamp
assert grader.validate_score(1.5)  == 1.0
assert grader.validate_score(-0.5) == 0.0
try:
    grader.validate_score("bad")
    assert False
except TypeError:
    pass
print("PASS 9: validate_score clamp + TypeError")

# 10. Unsupported task_id
try:
    grader.grade_episode("expert", [], {}, 0)
    assert False
except ValueError:
    pass
print("PASS 10: unknown task_id raises ValueError")

print("\nALL GRADER TESTS PASSED")
