from BackTranslation import BackTranslation
import csv
import random


def check_if_long_enough(original, aug, min_length=0.65):
    return len(aug) > min_length * len(original)


def check_difference(original, aug, difference=1):
    words_orig = original.split(' ')
    words_aug = aug.split(' ')
    count_differences = 0
    for i in range(len(words_orig)):
        if i < len(words_aug) and words_orig[i] != words_aug[i]:
            count_differences += 1
    return count_differences > difference


def back_translate_augment(original, trans):
    languages = ['es', 'de', 'sv', 'zh-cn']
    for _ in range(len(languages)):
        language = random.choice(languages)
        languages.remove(language)
        aug = trans.translate(original, src='en', tmp=language).result_text
        if check_difference(original, aug) and check_if_long_enough(original, aug):
            print('orignial:', original)
            print('aug:     ', aug)
            return aug


def main(threshold=0.8):
    trans = BackTranslation()
    with open('../parler-hate-speech-main/parler_annotated_data.csv', encoding="utf8") as file:
        csvreader = csv.reader(file, delimiter=',')
        next(csvreader)
        for row in csvreader:
            post = row[1]
            if random.random() < threshold:
                post = back_translate_augment(post, trans)


main()
