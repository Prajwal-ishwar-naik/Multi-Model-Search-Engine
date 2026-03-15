def fingers_up(lm_list):
    fingers = []

    if not lm_list:
        return fingers

    # Thumb
    fingers.append(lm_list[4][1] > lm_list[3][1])

    # Other fingers
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(lm_list[tip][2] < lm_list[tip - 2][2])

    return fingers
