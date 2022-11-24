from playsound import playsound

# mp3 files created from: https://ttsmp3.com/

sort_colour = {
    'glass': ['blue','bin'],
    'metal': ['blue','bin'],
    'plastic': ['blue','bin'],
    'trash': ['black','bin'],
    'cardboard':['yellow','bin'],
    'paper': ['yellow','bin'],
}

def play_sound(item_class):

    playsound(f'utils/mp3/{sort_colour[item_class][0]}.mp3')

    playsound(f'utils/mp3/{sort_colour[item_class][1]}.mp3')


play_sound('trash')