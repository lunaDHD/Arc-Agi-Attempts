# Arc-Agi-Attempts
Alright so this is pretty basic I know. Run ImgTrainer.py or main.py. I dont have a requirements list but make sure you have torch. I made a little reformatter script (reOrginize.py) to turn the json files into a better format for my usage. But don't worry about it it's just as follows:

```
[ # list of samples
    [ # single sample
        [ # list of examples
            [ # example / pair of input and output
                [], # 2d array input for example
                [] # 2d array output for example
            ]
        ],
        [ # pair of input and correct output
            [], # 2d array input for problem
            [] # 2d array correct output for problem
        ]
    ]
]
```
Yeah I know it's dumb and 5 deep and stupid but it's how i roll.