CITE_START = '[CITE]'
CITE_END = '[/CITE]'
INTENT_TOKENS = ['@BACK@', '@MOT@', '@FUT@', '@SIM@', '@DIF@', '@USE@', '@EXT@', '@UNSURE@']
SPECIAL_TOKENS = [CITE_START, CITE_END] + INTENT_TOKENS
IGNORE_SENTS = {'----------------------------------', '****'}


PAD_TOKEN_ID = -100