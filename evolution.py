import copy
import random
import numpy as np
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players, policy='top_k', save_learning_info=False):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        :param policy: policy for selection of players
        :param save_learning_info: plot learning curve of the players
        """
        policy_func = getattr(self, policy + '_selection')
        next_players = policy_func(players, num_players)

        if save_learning_info:
            # write learning info to a file
            sorted_next_players = sorted(next_players, key=lambda x: x.fitness, reverse=True)
            with open('learning_info.txt', 'a') as f:
                best_fit = sorted_next_players[0].fitness
                worst_fit = sorted_next_players[-1].fitness
                average_fit = sum([player.fitness for player in sorted_next_players]) / len(sorted_next_players)
                f.write(f'{best_fit},{worst_fit},{average_fit}\n')
        return next_players

    def generate_new_population(self, num_players, prev_players=None, policy='random'):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            policy_func = getattr(self, policy + '_selection')
            children = [self.cross_over(policy_func(prev_players, 1)[0], policy_func(prev_players, 1)[0]) for _ in range(num_players)]
            return [self.mutate(x) if random.random() < 0.2 else x for x in children]

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def cross_over(self, p1, p2):
        """
        Gets two players as an input and produces a child by crossing over the weights and biases of the two players.
        """
        best_p = p1 if p1.fitness > p2.fitness else p2
        worst_p = p1 if best_p == p2 else p2
        child = self.clone_player(best_p)
        for layer in child.nn.weights.keys():
            if layer % 2 == 0:
                child.nn.weights[layer] += worst_p.nn.weights[layer]
                child.nn.biases[layer] += worst_p.nn.biases[layer]
                child.nn.weights[layer] /= 2.
                child.nn.biases[layer] /= 2.

        child.fitness = (worst_p.fitness + best_p.fitness) / 2.  # An estimation of child fitness
        return child

    def mutate(self, player):
        """
        Gets a player as an input and produces a mutated player.
        """
        for layer in player.nn.weights.keys():
            player.nn.weights[layer] += np.random.normal(0, 0.1, player.nn.weights[layer].shape)
            player.nn.biases[layer] += np.random.normal(0, 0.1, player.nn.biases[layer].shape)
        return player

    def random_selection(self, players, num_players):
        """
        Gets list of players and returns num_players number of players based on random policy.
        """
        return [random.choice(players) for _ in range(num_players)]

    def top_k_selection(self, players, num_players):
        """
        Gets list of players and returns num_players number of players based on their fitness value.
        """
        players = sorted(players, key=lambda x: x.fitness, reverse=True)
        return players[:num_players]

    def roulette_selection(self, players, num_players):
        """
        Gets list of players and returns num_players number of players based on roulette wheel policy.
        """
        next_players = []
        total_fitness = sum([player.fitness for player in players])
        for _ in range(num_players):
            r = random.random() * total_fitness
            for player in players:
                r -= player.fitness
                if r <= 0:
                    next_players.append(player)
                    break
        return next_players

    def tournament_selection(self, players, num_players, Q=2):
        """
        Gets list of players and returns num_players number of players based on tournament policy.
        """
        next_players = []
        for _ in range(num_players):
            players_in_tournament = random.sample(players, Q)
            best_player = max(players_in_tournament, key=lambda x: x.fitness)
            next_players.append(best_player)
        return next_players
