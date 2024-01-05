import constances
import numpy as np


def calculateValue(company, gender, age, mood):
    return companies[company].calculate_value(gender, age, mood)


class CVCalculator:
    def __init__(self, value_weights: [], gender, age, mood, gender_off_value, age_off_value, mood_off_value):
        self.value_weighs = value_weights
        self.gender = gender

        self.age = 0
        for i in range(len(constances.AGE_CATEGORIES)):
            if age < constances.AGE_CATEGORIES[i]:
                break
            self.age += 1

        self.mood = mood

        self.mood_total_value = np.array(mood)
        self.mood_total_value = self.mood_total_value[np.argmax(self.mood_total_value)]

        self.gender_off_value = gender_off_value
        self.age_off_value = age_off_value
        self.mood_off_value = mood_off_value

    def calculate_value(self, gender, age, mood):
        points = 0
        if self.gender == gender:
            points += self.value_weighs[0]
        else:
            points += self.value_weighs[0] * self.gender_off_value

        if self.age == age:
            points += self.value_weighs[1]
        if abs(self.age - age) == 1:
            points += (self.value_weighs[1] * (1 - self.age_off_value)) / 2 + self.age_off_value * self.value_weighs[1]
        else:
            points += self.age_off_value * self.value_weighs[1]

        mood_sum = 0

        for i in range(len(constances.MOOD_CATEGORIES)):
            mood_sum += mood[i] * self.mood[i]

        mood_sum /= self.mood_total_value

        points += self.mood_off_value * self.value_weighs[2]
        points += self.value_weighs[2] * (1 - self.mood_off_value) * mood_sum

        return points


companies = {
    "Electronics": CVCalculator([0.5, 0.4, 0.1], 0, 29, [0, 0, 0, 4, 2, 0.5, 1], 0.4, 0.1, 0),
    "Drugstore": CVCalculator([0.6, 0.25, 0.15], 1, 35, [0, 0, 0, 4, 2, 0.5, 1], 0.2, 0.5, 0.1),
    "Vegan Food": CVCalculator([0.2, 0.6, 0.2], 1, 18, [0, 0, 0, 5, 1, 0, 1], 0.8, 0.1, 0.5),
    "Kiosk": CVCalculator([0.2, 0.3, 0.5], 1, 15, [0, 0, 0, 10, 4, 0, 4], 0.8, 0.4, 0.1)
}
