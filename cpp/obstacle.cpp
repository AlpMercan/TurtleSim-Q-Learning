#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "Q_learning.cpp"

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

bool check_target(const std::pair<int, int> &state)
{
    return (state.first >= 50 && state.first <= 90 && state.second >= 20 && state.second <= 100);
}

bool check_barricade(const std::pair<int, int> &state)
{

    return (state.first >= 25 && state.first <= 45 && state.second >= 30 && state.second <= 70);
}

void train_robot_with_obstacle(QLearningAgent &agent, int step_limit = 200)
{
    int success_count = 0;
    int fail_count = 0;

    for (int episode = 0; episode < 1000; ++episode)
    {
        std::pair<int, int> state = {0, 0};
        int total_reward_for_episode = 0;

        for (int step = 0; step < step_limit; ++step)
        {
            std::string action = agent.choose_action(state);
            std::pair<int, int> new_state = move_robot(state, action);
            int reward = -1;

            if (check_barricade(new_state))
            {
                reward = -100;
                agent.update_q_table(state, new_state, action, reward);
                fail_count++;
                break;
            }

            if (check_target(new_state))
            {
                reward = 100;
                success_count++;
            }

            total_reward_for_episode += reward;
            agent.update_q_table(state, new_state, action, reward);
            state = new_state;

            if (step == step_limit - 1)
            {

                agent.update_q_table(state, new_state, action, -100);
            }
        }

        agent.update_exploration_rate(episode);
    }

    std::cout << "Training complete with " << success_count << " successes and " << fail_count << " failures." << std::endl;

    std::ofstream file("q_table.json");
    nlohmann::json j = agent.get_q_table();
    file << j.dump();
    file.close();
}

int main()
{
    QLearningAgent agent;
    train_robot_with_obstacle(agent);
    return 0;
}
