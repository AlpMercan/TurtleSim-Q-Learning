#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>

struct State
{
    int x, y;

    bool operator==(const State &other) const
    {
        return x == other.x && y == other.y;
    }
};

namespace std
{
    template <>
    struct hash<State>
    {
        size_t operator()(const State &state) const
        {
            return hash<int>()(state.x) ^ hash<int>()(state.y);
        }
    };
}

class QLearningAgent
{
public:
    QLearningAgent(double lr = 0.1, double df = 0.5, double er = 1.0, double max_er = 1.0, double min_er = 0.01, double er_decay = 0.005, State target = {70, 60})
        : learning_rate(lr), discount_factor(df), exploration_rate(er), max_exploration_rate(max_er), min_exploration_rate(min_er), exploration_decay_rate(er_decay), target_state(target)
    {

        for (int x = -200; x <= 200; x++)
        {
            for (int y = -200; y <= 200; y++)
            {
                states.push_back({x, y});
            }
        }
        actions = {"up", "down", "left", "right"};

        for (auto &state : states)
        {
            for (auto &action : actions)
            {
                q_table[state][action] = 0;
            }
        }
    }

    int get_reward(const State &state)
    {
        return state == target_state ? 100 : -1;
    }

    void update_q_table(const State &state, const State &new_state, const std::string &action, int reward)
    {
        double max_future_q = -std::numeric_limits<double>::infinity();
        for (auto &a : actions)
        {
            max_future_q = std::max(max_future_q, q_table[new_state][a]);
        }

        double current_q = q_table[state][action];
        double new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q);
        q_table[state][action] = new_q;
    }

    std::string choose_action(const State &state)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);

        if (dis(gen) < exploration_rate)
        {
            std::uniform_int_distribution<> action_dis(0, actions.size() - 1);
            return actions[action_dis(gen)];
        }
        else
        {
            std::string best_action;
            double max_q = -std::numeric_limits<double>::infinity();
            for (auto &action : actions)
            {
                if (q_table[state][action] > max_q)
                {
                    max_q = q_table[state][action];
                    best_action = action;
                }
            }
            return best_action;
        }
    }

    State simulate_environment(const State &state, const std::string &action)
    {
        State new_state = state;
        if (action == "up")
        {
            new_state.y += 10;
        }
        else if (action == "down")
        {
            new_state.y -= 10;
        }
        else if (action == "left")
        {
            new_state.x -= 10;
        }
        else if (action == "right")
        {
            new_state.x += 10;
        }
        return new_state;
    }

    void train()
    {
        for (int episode = 0; episode < 1000; ++episode)
        {
            State state = {0, 0};
            bool done = false;

            while (!done)
            {
                std::string action = choose_action(state);
                State new_state = simulate_environment(state, action);
                int reward = get_reward(new_state);
                update_q_table(state, new_state, action, reward);
                state = new_state;

                if (state == target_state)
                {
                    done = true;
                }
            }

            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * exp(-exploration_decay_rate * episode);
        }
        std::cout << "Training complete" << std::endl;
    }

private:
    std::vector<State> states;
    std::vector<std::string> actions;
    std::unordered_map<State, std::unordered_map<std::string, double>> q_table;
    double learning_rate, discount_factor, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate;
    State target_state;
};

int main()
{
    QLearningAgent agent;
    agent.train();
    return 0;
}
