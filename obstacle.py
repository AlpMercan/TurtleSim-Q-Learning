import turtle
import random
import pickle
import matplotlib.pyplot as plt
from Q_learning import QLearningAgent

# Initialize cumulative rewards list
cumulative_rewards = []

# Initialize matplotlib for live plotting
plt.ion()
fig, ax = plt.subplots()
(line,) = ax.plot(cumulative_rewards)
plt.title("Cumulative Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")

colors = ["red", "green", "blue", "yellow", "purple", "orange"]


def move_turtle(action):
    x, y = robot.pos()
    if action == "up" and y < 190:
        robot.setheading(90)
        robot.forward(10)
    elif action == "down" and y > -190:
        robot.setheading(270)
        robot.forward(10)
    elif action == "left" and x > -190:
        robot.setheading(180)
        robot.forward(10)
    elif action == "right" and x < 190:
        robot.setheading(0)
        robot.forward(10)


def draw_target():
    target.penup()
    target.goto(50, 100)
    target.pendown()
    for _ in range(2):
        target.forward(40)
        target.right(90)
        target.forward(80)
        target.right(90)
    target.penup()


def check_target():
    x, y = robot.pos()
    if 50 <= x <= 90 and 20 <= y <= 100:
        return True
    return False


def draw_barricade():
    barricade.penup()
    # Position the barricade between the reset point and the target
    barricade.goto(25, 50)
    barricade.pendown()
    for _ in range(2):
        barricade.forward(20)  # Width of the barricade
        barricade.right(90)
        barricade.forward(40)  # Height of the barricade
        barricade.right(90)
    barricade.penup()


def check_barricade():
    x, y = robot.pos()
    # Adjust these coordinates according to the new barricade position
    if 25 <= x <= 45 and 30 <= y <= 70:
        return True
    return False


def train_robot(agent, step_limit=200):
    success_count = 0
    fail_count = 0

    for episode in range(1000):
        total_reward_for_episode = 0
        state = (0, 0)
        robot.goto(0, 0)
        color = random.choice(colors)
        robot.pencolor(color)

        for step in range(step_limit):
            action = agent.choose_action(state)
            move_turtle(action)
            new_state = (int(robot.xcor()), int(robot.ycor()))
            print(
                f"Episode: {episode}, Successes: {success_count}, Fail: {fail_count},  Step: {step}, State: {state}, Action: {action}, New State: {new_state}"
            )

            if check_barricade():
                reward = -50
                agent.update_q_table(state, new_state, action, reward)
                robot.goto(0, 0)
                robot.clear()
                state = (0, 0)
                fail_count += 1
                print(f"failed! Episode: {episode}, Fail: {fail_count},Step: {step}")
                break

            if check_target():
                reward = 100
                agent.update_q_table(state, new_state, action, reward)
                success_count += 1
                total_reward_for_episode += reward
                print(
                    f"Success! Episode: {episode}, Step: {step}, Total Successes: {success_count}"
                )
                break
            else:
                reward = -1
                total_reward_for_episode += reward

            agent.update_q_table(state, new_state, action, reward)
            state = new_state

            if step == step_limit - 1:
                reward = -100
                agent.update_q_table(state, new_state, action, reward)
                robot.goto(0, 0)
                state = (0, 0)

        robot.clear()

        cumulative_rewards.append(total_reward_for_episode)
        line.set_xdata(range(len(cumulative_rewards)))
        line.set_ydata(cumulative_rewards)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        agent.exploration_rate = max(
            agent.min_exploration_rate,
            agent.exploration_rate * agent.exploration_decay_rate,
        )

    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

    print(f"Training complete with {success_count} successes.")


window = turtle.Screen()
window.title("TurtleSim Robot Simulation")
window.screensize(canvwidth=400, canvheight=400)
window.setup(width=500, height=500)

robot = turtle.Turtle()
robot.shape("turtle")

target = turtle.Turtle()
target.color("blue")
target.hideturtle()
draw_target()

barricade = turtle.Turtle()
barricade.color("grey")
barricade.hideturtle()
draw_barricade()

agent = QLearningAgent()
train_robot(agent)

window.mainloop()
