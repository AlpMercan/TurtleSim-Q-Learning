#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "Q_learning.cpp"

// Function to move the robot in the simulated environment
std::pair<int, int> move_robot(const std::pair<int, int> &state, const std::string &action)
{
    int x = state.first, y = state.second;
    if (action == "up" && y < 190)
        y += 10;
    else if (action == "down" && y > -190)
        y -= 10;
    else if (action == "left" && x > -190)
        x -= 10;
    else if (action == "right" && x < 190)
        x += 10;
    return {x, y};
}

// Function to check if the robot reached the target
bool check_target(const std::pair<int, int> &state)
{
    return (state.first >= 50 && state.first <= 90 && state.second >= 20 && state.second <= 100);
}

// Training function
void train_robot(QLearningAgent &agent, int step_limit = 200)
{
    int success_count = 0;

    for (int episode = 0; episode < 1000; ++episode)
    {
        std::pair<int, int> state = {0, 0}; // Starting position
        int total_reward_for_episode = 0;

        for (int step = 0; step < step_limit; ++step)
        {
            std::string action = agent.choose_action(state);
            std::pair<int, int> new_state = move_robot(state, action);
            int reward = -1; // Negative reward for each step

            if (check_target(new_state))
            {
                reward = 100;
                success_count++;
                break;
            }

            total_reward_for_episode += reward;
            agent.update_q_table(state, new_state, action, reward);
            state = new_state;

            if (step == step_limit - 1)
            {
                // Additional punishment for not reaching the target in time
                agent.update_q_table(state, new_state, action, -100);
            }
        }

        // Update exploration rate in the agent
        agent.update_exploration_rate(episode);
    }

    std::cout << "Training complete with " << success_count << " successes." << std::endl;

    // Save the Q-table to a file
    std::ofstream file("q_table.json");
    nlohmann::json j = agent.get_q_table(); // Assuming QLearningAgent has a method to return Q-table in JSON format
    file << j.dump();
    file.close();
}

int main()
{
    QLearningAgent agent;
    train_robot(agent);
    return 0;
}
