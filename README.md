# ComfyUI-KeySyncWrapper
WIP implementation of KeySync in ComfyUI

You'll need to git clone the original KeySync code into the KeySyncWrapper folder where this node sits. 
Follow the instructions in the keysync repo to install the pretrained models. They should go in a separate folder called pretrained_models (see below). 



Then everything should work? Maybe?
Lots of dependencies so beware installation issues. 

Reach out if you can help me clean up the code!

```bash
cd ../ComfyUI/custom_nodes/ComfyUI-KeySyncWraper
git clone https://github.com/antonibigata/keysync
```

```
custom_nodes/ComfyUI-KeySyncWrapper
  ├── keysync/
  │   ├── [KeySync code from KeySync repo https://github.com/antonibigata/keysync]
  ├── pretrained_models/
  │   ├── interpolation_dub.pt
  │   └── keyframe_dub.pt
  ├── __init__.py
  ├── infer.py
  ├── nodes.py
```
