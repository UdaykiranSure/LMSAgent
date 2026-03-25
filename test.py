from datetime import date
from mistralai import Mistral
from utils import format_messages_for_mistral

from environment import StudentAgentEnvironment
from mockdb import MockStudentDB
from oracle import Oracle
from reward import reward  # your function

# 1. Setup
today = date.today()
query = "What is the deadline for RL assignment?"

db = MockStudentDB(today)
oracle = Oracle()
gt = oracle.derive(query, db, today)

env = StudentAgentEnvironment(db, gt, query, today)

# 2. Mistral client
client = Mistral(api_key="LsFhKFFKJBmYjNnfVC8PRsab1uf5ZhlN")

# 3. Run episode
done = False
a = 1
while not done:
    messages = env.state.context_window
    print(a, messages,  )
    print('##'*90)
    messages = format_messages_for_mistral(env.state.context_window)

    response = client.chat.complete(
        model="mistral-small-latest",  # or mistral-medium
        messages=messages,
        temperature=0.2
    )

    agent_output = response.choices[0].message.content
    tool_calls = response.choices[0].message.tool_calls
    obs = env.step(agent_output)

    done = obs["done"]
    a += 1
# 4. Build trajectory
traj = env.build_trajectory()
print(traj["tool_calls"])
# 5. Compute reward
score = reward(traj, gt)

print("Final Score:", score)
print("Final Response:", traj["final_response"])


