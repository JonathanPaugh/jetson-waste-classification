from os.path import join

from playsound import playsound

SOUND_DIRECTORY = 'sound'
SOUND_EXTENSION = '.mp3'

sound_map = {
    'glass': ['blue', 'bin'],
    'metal': ['blue', 'bin'],
    'plastic': ['blue', 'bin'],
    'trash': ['black', 'bin'],
    'cardboard': ['yellow', 'bin'],
    'paper': ['yellow', 'bin'],
}

def play_sound(class_name):
    color, container = sound_map[class_name]
    playsound(join(SOUND_DIRECTORY, f'{color}{SOUND_EXTENSION}'))
    playsound(join(SOUND_DIRECTORY, f'{container}{SOUND_EXTENSION}'))
