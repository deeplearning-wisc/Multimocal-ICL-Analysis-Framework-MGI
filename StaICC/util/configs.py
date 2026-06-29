STANDARD_SETTINGS = {
    "calibration_number": 50,
    "demonstration_number": 50,
    "test_number": 580,
    "test_times_for_each_test_sample": 1,
    "cut_by_length_remain_short": 1024,
    "cut_by_length_remain_long": 8192,
    "ece_bins": 10,
    "random_seed": 42,
    "random_A": 1664525,
    "random_B": 1013904223,
    "random_C": 2**32,
    "split_for_FP": {
        "calibration_number": 1024,
        "demonstration_number": 512,
        "test_number": 512
    },
    "split_for_TEE": {
        "calibration_number": 200,
        "demonstration_number": 300,
        "test_number": 512
    },
    "L9(3,4)_orthogonal_table": [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 2, 2, 2],
        [1, 0, 1, 2],
        [1, 1, 2, 0],
        [1, 2, 0, 1],
        [2, 0, 2, 1],
        [2, 1, 0, 2],
        [2, 2, 1, 0]
    ]
}

STRICT_MODE = True

WARNING_SETTINGS = {
    "tampering": "You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.",
    "FP_length_warning": "We are spliting the financial_phrasebank with a shorter dataset length. The default spliting can't be remained. Be careful.",
    "basic_dataset_template_protect": "You are editing the basic dataset template in the strict mode. Canceled.\n If you want to edit the prompt template, please edit the dataset_interface.prompt_writter.",
    "strict_mode_protect": "The setting can't be changed in the strict mode. Return to default."
}

PPL_ICL_INSTRUCTION_SETTINGS = {
    # https://aclanthology.org/2023.findings-emnlp.679.pdf
    # Use GPT-4 for paraphrase
    "GLUE-SST2": [
        "How would you describe the overall feeling of the movie based on this sentence? ",
        "What mood does this sentence convey about the movie? ",
        "What is the reviewer's opinion of the movie? ",
        "What are the reviewer's thoughts on the movie? ",
        "What impression does the reviewer have of the movie? "
    ],
    "rotten_tomatoes": [
        "How would you describe the overall feeling of the movie based on this sentence? ",
        "What mood does this sentence convey about the movie? ",
        "What is the reviewer's opinion of the movie? ",
        "What are the reviewer's thoughts on the movie? ",
        "What impression does the reviewer have of the movie? "
    ],
    "financial_phrasebank": [
        "What is the attitude towards the financial news in this sentence? ",
        "What is the emotional response to the financial news in this sentence? ",
        "What is the reaction to the financial news in this sentence? ",
        "How does this sentence convey feelings about the financial news? ",
        "How is the financial news perceived in this sentence? "
    ],
    "SST5": [
        "How would you describe the overall feeling of the movie based on this sentence? ",
        "What mood does this sentence convey about the movie? ",
        "What is the reviewer's opinion of the movie? ",
        "What are the reviewer's thoughts on the movie? ",
        "What impression does the reviewer have of the movie? "
    ],
    "TREC": [
        "What is the topic of the question? ",
        "What is the primary focus of this question? ",
        "What is the central subject of this question? ",
        "What does the question concern? ",
        "What is the question addressing? "
    ],
    "AGNews": [
        "What is the topic of the news? ",
        "What is the news focused on? ",
        "What is the subject of the news? ",
        "What does the news cover? ",
        "What is the news discussing? "
    ],
    "Subjective": [
        "Does this sentence reflect a personal opinion? ",
        "Is this sentence expressing a personal opinion or stating a fact? ",
        "Is this sentence based on personal opinion or factual information? ",
        "Is this sentence expressing a personal view or a factual statement? ",
        "Is this sentence expressing a personal perspective or presenting objective facts? "
    ],
    "tweet_eval_emotion": [
        "What feeling does this sentence convey? ",
        "What emotion does this sentence express? ",
        "What emotion is conveyed by this sentence? ",
        "What is the emotional tone of this sentence? ",
        "What is the mood reflected in this sentence? "
    ],
    "tweet_eval_hate": [
        "Does this sentence contain hate speech? ",
        "Is this sentence an example of hate speech? ",
        "Does this sentence convey hateful language? ",
        "Is this sentence indicative of hate speech? ",
        "Does this sentence involve hate speech? "
    ],
    "hate_speech_18": [
        "Does this sentence contain hate speech? ",
        "Is this sentence an example of hate speech? ",
        "Does this sentence convey hateful language? ",
        "Is this sentence indicative of hate speech? ",
        "Does this sentence involve hate speech? "
    ],
}