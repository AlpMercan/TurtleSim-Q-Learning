import turtle
import tkinter as tk
from Q_learning import (
    QLearningAgent,
)
import random
import pickle


def move_turtle(action):
    x, y = robot.pos()

    if (
        action == "up" and y < 190
    ):  # Slightly less than the boundary to prevent overshooting
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
    target.goto(50, 100)  # Top-left corner of the rectangle
    target.pendown()
    for _ in range(2):
        target.forward(40)  # Width of the rectangle
        target.right(90)
        target.forward(80)  # Height of the rectangle
        target.right(90)
    target.penup()


def check_target():
    x, y = robot.pos()
    if 50 <= x <= 90 and 20 <= y <= 100:
        return True
    return False


colors = ["red", "green", "blue", "yellow", "purple", "orange"]


def train_robot(agent, step_limit=200):
    success_count = 0  # Counter for successful episodes

    for episode in range(1000):
        state = (0, 0)  # Starting position
        robot.goto(0, 0)
        color = random.choice(colors)
        robot.pencolor(color)

        for step in range(step_limit):
            action = agent.choose_action(state)
            move_turtle(action)

            new_state = (int(robot.xcor()), int(robot.ycor()))
            print(
                f"Episode: {episode}, Successes: {success_count}, Step: {step}, State: {state}, Action: {action}, New State: {new_state}"
            )

            if check_target():
                reward = 100  # Positive reward for reaching the target
                agent.update_q_table(state, new_state, action, reward)
                success_count += 1
                print(
                    f"Success! Episode: {episode}, Step: {step}, Total Successes: {success_count}"
                )
                break
            else:
                reward = -1  # Negative reward for each step

            # Update the Q-table
            agent.update_q_table(state, new_state, action, reward)
            state = new_state

            if step == step_limit - 1:
                reward = (
                    -100
                )  # Additional punishment for not reaching the target in time
                agent.update_q_table(state, new_state, action, reward)
                robot.goto(0, 0)  # Reset turtle's position
                state = (0, 0)  # Also reset the state

        # Update exploration rate in the agent
        agent.exploration_rate = max(
            agent.min_exploration_rate,
            agent.exploration_rate * agent.exploration_decay_rate,
        )

    # Save the Q-table to a file after training is complete
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

    print(f"Training complete with {success_count} successes.")
    window.textinput(
        "Training Complete",
        "The training process has finished. Total successes: " + str(success_count),
    )


# Setting up the screen and turtle
window = turtle.Screen()
window.title("TurtleSim Robot Simulation")
window.screensize(canvwidth=400, canvheight=400)  # Adjust the drawing area size
window.setup(width=500, height=500)

robot = turtle.Turtle()
robot.shape("turtle")

target = turtle.Turtle()
target.color("blue")
target.hideturtle()
draw_target()

# Initialize QLearningAgent and start training
agent = QLearningAgent()
train_robot(agent)

window.mainloop()
