import back_translate
import csv
import random
import BackTranslation
from back_translate import back_translate_augment

def main(threshold=0.8):
    trans = BackTranslation()
    with open('parler-hate-speech-main/parler_annotated_data.csv', encoding="utf8") as file:
        csvreader = csv.reader(file, delimiter=',')
        next(csvreader)
        for row in csvreader:
            post = row[1]
            if random.random() < threshold:
                post = back_translate_augment(post, trans)


main()
