CITE_START = '[CITE]'
CITE_END = '[/CITE]'
INTENT_TOKENS = ['@BACK@', '@MOT@', '@FUT@', '@SIM@', '@DIF@', '@USE@', '@EXT@', '@UNSURE@']
SPECIAL_TOKENS = [CITE_START, CITE_END] + INTENT_TOKENS
IGNORE_SENTS = {'----------------------------------', '****'}


PAD_TOKEN_ID = -100


WRONG_INTENT_INSTANCE_START_ID = 30000      # note that the gold instances can go up to id=12571
CLIPPED_INSTANCE_START_ID = 300000          # consider num_gold_instances X all intents can go up to id=100k
