#include <iostream>
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

std::string choose_best_action(const std::pair<int, int> &state, const nlohmann::json &q_table)
{
    std::string best_action;
    double max_q = -std::numeric_limits<double>::infinity();

    for (auto &action : q_table[std::to_string(state.first)][std::to_string(state.second)].items())
    {
        if (action.value().get<double>() > max_q)
        {
            max_q = action.value().get<double>();
            best_action = action.key();
        }
    }

    return best_action;
}

void run_simulation(const std::string &q_table_path = "q_table.json")
{

    std::ifstream file(q_table_path);
    nlohmann::json q_table;
    file >> q_table;
    file.close();

    std::pair<int, int> state = {0, 0};

    while (!check_target(state))
    {
        std::string action = choose_best_action(state, q_table);
        state = move_robot(state, action);
    }

    std::cout << "Target reached!" << std::endl;
}

int main()
{
    run_simulation();
    return 0;
}
