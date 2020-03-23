# CME241

This repository is used for managing my assignment solution as well as study materials for [CME 241: Reinforcement Learning for Stochastic Control Problems in Finance](http://web.stanford.edu/class/cme241/), taught by Prof. Ashwin Rao at Stanford University, winter 2020.

---

## Assignment List

My solution for assignments are listed and linked as follows:

| Lecture |                           Topic                            |                 Written Assignment                 | Programming Assignment                                                                                                                                                                                    |
| ------- | :--------------------------------------------------------: | :------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1       |                          Overview                          |                                                    |
| 2       |                      MP, MRP and MDP                       |          [Link](./assignment/writeup/lecture_2.pdf)           | [MP](./assignment/src/process/mp.py) / [MRP](./assignment/src/process/mrp.py) / [MDP](./assignment/src/process/mdp.py) / [Policy](./assignment/src/process/policy.py)                                                                                 |
| 3       |                    Dynamic Programming                     |          [Link](./assignment/writeup/lecture_12.pdf)          | [Policy Evaluation](./assignment/src/algorithm/policy_eval.py) / [Policy Iteration](./assignment/src/algorithm/pi.py) / [Value Iteration](./assignment/src/algorithm/vi.py)                                                                |
| 4       |               Risk-Aversion, Utility Theory                |          [Link](./assignment/writeup/lecture_4.pdf)           |                                                                                                                                                                                                           |
| 5-9     |           Application Problems of RL in Finance            | [Merton's Portfolio problem](./assignment/writeup/merton.pdf) | [Optimal Asset Allocation](./assignment/src/example/merton.py)                                                                                                                                                       |
| 10-11   |                   Model-free Prediction                    |                                                    | [Interface](./assignment/src/tabular_rl_interface.py) / [Monte-Carlo](./assignment/src/algorithm/mc.py) / [TD(0)](./assignment/src/algorithm/td.py) / [TD(lambda)](./assignment/src/algorithm/td_lambda.py) / [Comparison](./assignment/src/example/mc_vs_td.py) |
| 12      |                     Model-free Control                     |          [Link](./assignment/writeup/lecture_12.pdf)          | [MC Control](./assignment/src/algorithm/mc.py) / [SARSA](./assignment/src/algorithm/sarsa.py) / [Q-Learning](./assignment/src/algorithm/q_learning.py)                                                                                     |
| 13-14   |                   Function Approximation                   |                                                    | TBD                                                                                                                                                                                                       |
| 15      |            Value Function Geometry, Gradient TD            |                                                    |
| 16      |                       Guest Lecture                        |                                                    |
| 17      |                      Policy Gradient                       |          [Link](./assignment/writeup/lecture_17.pdf)          | [REINFORCE](./assignment/src/algorithm/reinforce.py)                                                                                                                                                                 |
| 18      | Evolutionary Strategies, Integrating Learning and Planning |                                                    |
| 19      |                Exploration vs Exploitation                 |               [Link](./assignment/src/img/mab)                | [Multi-armed Bandits](./assignment/src/example/mab.py)                                                                                                                                                               |
| 20      |                       Special Topics                       |                                                    |

To install all dependencies, run:

`pip install -r ./assignment/requirements.txt`

## Resources

A list of resources (including github repo, lectures, papers, talks etc.) which I personally find useful are listed and linked [here](./assignment/resources/README.md)
