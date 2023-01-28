import random
from back_translate.translation import BackTranslation
from back_translate.translation_Baidu import BackTranslation_Baidu


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


def back_translate_augment(original, trans, validity_check=False, p=0.8, verbose=0):
    languages = ['es', 'de', 'sv', 'zh-cn']
    if random.random() < p:
        if validity_check:
            for _ in range(len(languages)):
                language = random.choice(languages)
                languages.remove(language)
                aug = trans.translate(original, src='en', tmp=language).result_text
                if check_difference(original, aug) and check_if_long_enough(original, aug):
                    break
        else:
            language = random.choice(languages)
            aug = trans.translate(original, src='en', tmp=language).result_text
        if verbose > 0:
            print('orignial:', original)
            print('aug:     ', aug)
        return aug
    return original