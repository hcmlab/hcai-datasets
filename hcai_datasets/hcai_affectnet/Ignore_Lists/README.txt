Use this ignore lists to exclude filed from training that you do not want.
To do this create a respective builder config in hcai_affectnet.py.
At the moment only the 'ignore_duplicate_list' and 'ignore_list_wrong_format' is used.

ignore_list_duplicates:         	Ignores duplicate files found in the sets
ignore_list_wrong_format: 		Ignores all files that are not in tensorflow compatible encodings. I.e. not jpg, png or bmp

# Not available
#affectnet_ignore_list_5000: 		Ignore all files that are not used in this repo https://github.com/ankitsharma285/AffectNet/tree/32a98f4b215221a4738484e5ec8e8836b6d62949 to enable better comparison
#affectnet_ignore_list_valence_arousal:	Ignore all files that have no annotated valence and arousal values
