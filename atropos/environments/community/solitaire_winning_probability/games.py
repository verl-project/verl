import random


def easy_game_1():
    """
    Draw a card from a deck of 4 cards (1, 2, 3, 4). Win if the card is 1.
    """
    deck = [1, 2, 3, 4]
    random.shuffle(deck)
    return deck[0] == 1


def easy_game_2():
    """
    Roll a 6-sided die. Win if the result is 1.
    """
    return random.randint(1, 6) == 1


def easy_game_3():
    """
    Flip a coin twice. Win if both are heads. (Assuming 0 for heads, 1 for tails)
    """
    flip1 = random.randint(0, 1)  # 0 for heads, 1 for tails
    flip2 = random.randint(0, 1)
    return flip1 == 0 and flip2 == 0


def easy_game_4():
    """
    Draw a card from a deck of 3 cards (1, 2, 3). Win if the card is 1.
    """
    deck = [1, 2, 3]
    random.shuffle(deck)
    return deck[0] == 1


def card_matching_game_1():
    """
    A game where we lose if the counter matches the card value.

    Rules:
    1. We have a standard deck where each rank (1-13) appears 4 times (52 cards total)
    2. Cards are shuffled randomly
    3. We deal cards one by one, keeping a counter that cycles 1,1,1,1,1,1,1,1,...
    4. We lose if the counter matches the card value
    5. We win if we go through the whole deck without any matches

    Returns:
        bool: True if won, False if lost
    """
    # Create and shuffle deck
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    deck = [rank for rank in ranks for _ in range(1)]
    random.shuffle(deck)

    # Play game
    count = 0
    for card in deck:
        count = (count % 1) + 1
        if count == card:
            return False
    return True


def card_matching_game_2():
    """
    A game where we lose if the counter matches the card value.

    Rules:
    1. We have a standard deck where each rank (1-13) appears 4 times (52 cards total)
    2. Cards are shuffled randomly
    3. We deal cards one by one, keeping a counter that cycles 1,2,1,2,1,2,1,2,...
    4. We lose if the counter matches the card value
    5. We win if we go through the whole deck without any matches

    Returns:
        bool: True if won, False if lost
    """
    # Create and shuffle deck
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    deck = [rank for rank in ranks for _ in range(4)]
    random.shuffle(deck)

    # Play game
    count = 0
    for card in deck:
        count = (count % 2) + 1
        if count == card:
            return False
    return True


def card_matching_game_3():
    """
    A game where we lose if the counter matches the card value.

    Rules:
    1. We have a standard deck where each rank (1-13) appears 4 times (52 cards total)
    2. Cards are shuffled randomly
    3. We deal cards one by one, keeping a counter that cycles 1,2,3,1,2,3,...
    4. We lose if the counter matches the card value
    5. We win if we go through the whole deck without any matches

    Returns:
        bool: True if won, False if lost
    """
    # Create and shuffle deck
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    deck = [rank for rank in ranks for _ in range(4)]
    random.shuffle(deck)

    # Play game
    count = 0
    for card in deck:
        count = (count % 3) + 1
        if count == card:
            return False
    return True


def card_matching_game_4():
    """
    A game where we lose if the counter matches the card value.

    Rules:
    1. We have a standard deck where each rank (1-13) appears 4 times (52 cards total)
    2. Cards are shuffled randomly
    3. We deal cards one by one, keeping a counter that cycles 1,2,3,4,1,2,3,4,...
    4. We lose if the counter matches the card value
    5. We win if we go through the whole deck without any matches

    Returns:
        bool: True if won, False if lost
    """
    # Create and shuffle deck
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    deck = [rank for rank in ranks for _ in range(4)]
    random.shuffle(deck)

    # Play game
    count = 0
    for card in deck:
        count = (count % 4) + 1
        if count == card:
            return False
    return True


def odd_card_game():
    """
    A game where we win if we draw an odd-valued card from a deck.

    Rules:
    1. We have a standard deck where each rank (1-13) appears 4 times (52 cards total)
    2. Cards are shuffled randomly
    3. We draw one card randomly from the deck
    4. We win if the card value is odd, lose if it's even

    Returns:
        bool: True if won (odd card drawn), False if lost (even card drawn)
    """
    # Create and shuffle deck
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    deck = [rank for rank in ranks for _ in range(4)]
    random.shuffle(deck)

    # Draw one card and check if it's odd
    drawn_card = deck[0]
    return drawn_card % 2 == 1
