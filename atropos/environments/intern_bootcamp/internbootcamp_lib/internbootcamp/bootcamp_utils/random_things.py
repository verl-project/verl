import random
with open('./internbootcamp/libs/data/words_alpha_370000.txt', 'r') as f:
    words = f.readlines()
    words = [w.strip() for w in words]
def random_word():
    w = random.choice(words)
    while len(w) < 3:
        w = random.choice(words)
    return w


if __name__ == '__main__':
    print(random_word())