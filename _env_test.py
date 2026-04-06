import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./server")

from server.environment import SREEnvironment
from models import Action, ActionType

env = SREEnvironment()

for task_id in ["easy", "medium", "hard"]:
    obs = env.reset(task_id)
    task = env._sim._task_info
    print(f"\n{'='*60}")
    print(f"TASK: {task_id} | name={task.name} | max_ticks={task.max_ticks}")
    print(f"DESC: {task.description[:100]}...")

    obs = env.reset(task_id)
    print(f"reset OK | tick={obs.tick} | site_up={obs.site_uptime} | alerts={len(obs.alerts)}")

    # Verify task_description propagates correctly
    assert task.description in obs.task_description or obs.task_description == task.description, \
        f"task_description mismatch: {obs.task_description!r}"

    # Verify action_history is empty after reset
    assert len(env._sim.action_history) == 0, "action_history not cleared on reset"

    # Verify scenario setups
    if task_id == "easy":
        web3 = obs.servers.get("web-3")
        assert web3 is not None and web3.status == "offline", \
            f"easy: web-3 should be offline, got {web3.status if web3 else 'None'}"
        print(f"  easy: web-3 offline={web3.status}")

    elif task_id == "medium":
        gw1 = obs.servers.get("api-gw-1")
        assert gw1 is not None and gw1.cpu >= 85, \
            f"medium: api-gw-1 cpu should be high, got {gw1.cpu if gw1 else 'None'}"
        print(f"  medium: api-gw-1 cpu={gw1.cpu:.1f}%")

    elif task_id == "hard":
        w1 = obs.servers.get("web-1")
        assert w1 is not None and w1.ram >= 70, \
            f"hard: web-1 ram should be leaking, got {w1.ram if w1 else 'None'}"
        print(f"  hard: web-1 ram={w1.ram:.1f}% | version={w1.version}")

    # Take one step
    if task_id == "easy":
        act = Action(action_type=ActionType.RestartService, target_id="web-3")
    elif task_id == "medium":
        act = Action(action_type=ActionType.ScaleUp, target_id="api-gw-1")
    else:
        act = Action(action_type=ActionType.InvestigateLog, target_id="web-1")

    resp = env.step(act)
    print(f"step OK | score={resp.reward.score:.3f} | done={resp.done}")
    print(f"  feedback: {resp.reward.feedback[:80]}")

    # Verify done fires when site goes down
    env2 = SREEnvironment()
    env2.reset(task_id)
    # Force site down
    for srv in env2._sim.state["servers"].values():
        srv.status = type(srv.status).offline if hasattr(srv.status, 'offline') else "offline"
        from models import ServerStatus
        srv.status = ServerStatus.offline
    resp2 = env2.step(Action(action_type=ActionType.InvestigateLog, target_id="web-1"))
    assert resp2.done, f"done should be True when site is down, got done={resp2.done}"
    print(f"  done-on-site-down: OK (done={resp2.done})")

print("\n" + "="*60)
print("ALL CHECKS PASSED")
