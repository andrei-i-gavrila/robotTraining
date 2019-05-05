from collections import defaultdict

import numpy as np
from math import floor
from numpy import random


class Rule:

    def __init__(self, conditions, result):
        self.conditions = conditions
        self.result = result
        self.numerosity = 1
        self.__match_count = 0
        self.correct_count = 0
        self.accuracy_value = None
        self.fitness_value = None

    @property
    def match_count(self):
        return self.__match_count

    @match_count.setter
    def match_count(self, count):
        self.__match_count = count
        self.fitness_value = None
        self.accuracy_value = None
        self.fitness()

    def accuracy(self):
        if self.accuracy_value is None:
            self.accuracy_value = self.correct_count / self.match_count
        return self.accuracy_value

    def fitness(self):
        if self.fitness_value is None:
            self.fitness_value = self.accuracy() ** 2
        return self.fitness_value

    def mutate(self, data, mutation_rate=0.2):
        for i in range(len(data)):
            if random.random() < mutation_rate:
                self.conditions[i] = data[i]

    def matches(self, data):
        return all(self.conditions[i] >= data[i] for i in range(len(data)))

    def subsumes(self, rule):
        if self == rule or rule.accuracy() > self.accuracy():
            return False
        if any(self.conditions[i] < rule.conditions[i] for i in range(len(self.conditions))):
            return False

        self.numerosity += rule.numerosity
        return True


def crossover(rule1: Rule, rule2: Rule):
    for i in range(len(rule1.conditions)):
        if rule1.conditions[i] != rule2.conditions[i] and random.random() < 0.5:
            rule1.conditions[i], rule2.conditions[i] = rule2.conditions[i], rule1.conditions[i]


def get_matching_rules(rules, data, answer):
    matching = filter(lambda r: r.matches(data), rules)
    correct, incorrect = [], []
    for rule in matching:
        (incorrect, correct)[rule.result == answer].append(rule)

    return correct, incorrect


def cover(data, answer, covering_rate=0.3):
    cover = [1 if random.random() < covering_rate else 1000 for _ in range(len(data) - 1)] + [1]
    random.shuffle(cover)
    return Rule(np.multiply(data, cover), answer)


def update_rule_params(correct_match, incorrect_match):
    for rule in correct_match:
        rule.match_count += 1
        rule.correct_count += 1
    for rule in incorrect_match:
        rule.match_count += 1


def subsume(rules):
    new_rules = []
    for rule1 in rules:
        if not any(1 for r in rules if r.subsumes(rule1)):
            new_rules.append(rule1)
    return new_rules


def evolve(data, rules, tournament_size=5):
    if len(rules) < tournament_size:
        return set()

    parent1 = max(random.choice(rules, tournament_size), key=lambda r: r.fitness())
    parent2 = max(random.choice(rules, tournament_size), key=lambda r: r.fitness())
    offspring1 = Rule([condition for condition in parent1.conditions], parent1.result)
    offspring2 = Rule([condition for condition in parent2.conditions], parent2.result)

    offspring1.mutate(data)
    offspring2.mutate(data)

    crossover(offspring1, offspring2)

    offspring1.mutate(data)
    offspring2.mutate(data)

    offspring1.match_count = 1
    offspring1.correct_count = 1
    offspring2.match_count = 1
    offspring2.correct_count = 1

    return [offspring1, offspring2]


def deletion(rules):
    rules.sort(key=lambda r: r.fitness(), reverse=True)
    count = sum(r.numerosity for r in rules)
    while count > max_rules_count:
        count -= rules[-1].numerosity
        del rules[-1]
    return rules


def train(training_data):
    rules = []
    count = 0
    for data, answer in training_data:
        count += 1
        if count % 100 == 0:
            print("processed ", count, "entries")
        correct_rules, incorrect_rules = get_matching_rules(rules, data, answer)
        if len(correct_rules) == 0:
            new_rule = cover(data, answer)
            rules.append(new_rule)
            correct_rules = {new_rule}

        update_rule_params(correct_rules, incorrect_rules)
        rules = subsume(rules)
        rules.extend(evolve(data, correct_rules))

        rules = deletion(rules)

    for rule in rules:
        for i in range(len(rule.conditions)):
            if rule.conditions[i] > 5.0:
                rule.conditions[i] = 6.0
    return rules


def test(test_data, rules):
    correct_answer_count = 0
    count = 0
    for data, answer in test_data:
        count += 1
        matching_rules = list(filter(lambda r: r.matches(data), rules))

        result = defaultdict(float)

        for rule in matching_rules:
            result[rule.result] += rule.numerosity * rule.fitness()

        if len(result) == 0:
            continue

        if max(result.keys(), key=lambda r: result[r]) == answer:
            correct_answer_count += 1

        if count % 100 == 0:
            print("Accuracy", correct_answer_count / count)

    return correct_answer_count / count


def read_data(file, training_rate=0.7):
    with open(file) as f:
        lines = list(map(lambda l: l.strip().split(','), f.readlines()))
        # shuffle(lines)
        input_data = [list(map(float, line[:-1])) for line in lines]
        answers = [line[-1] for line in lines]

        training_cutoff = floor(len(input_data) * training_rate)
        training_data = list(zip(input_data[:training_cutoff], answers[:training_cutoff]))
        test_data = list(zip(input_data[training_cutoff:], answers[training_cutoff:]))

    return training_data, test_data


if __name__ == "__main__":
    random.seed(1)
    max_rules_count = 500

    training_data, test_data = read_data('data4.txt')

    rules = train(training_data)

    accuracy = test(test_data, rules)

    print("Accuracy", accuracy)
