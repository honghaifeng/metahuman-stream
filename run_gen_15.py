import torch
_original_load = torch.load
def patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

import sys
sys.argv = ['genavatar.py', '--file', 'data/video/15_clip.mp4', '--avatar_id', 'musetalk_15', '--version', 'v15']

exec(open('avatars/musetalk/genavatar.py').read())
