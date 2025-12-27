def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.
    This is a simple heuristic which truncates the longer sequence first.
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """
    Converts a single InputExample into BERT-style features:
    input_ids, input_mask, segment_ids, original_tokens
    """

    # Tokenize input text
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None

    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    # Account for [CLS], [SEP], [SEP]
    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: max_seq_length - 2]

    # Segment IDs
    seg_id_a = 0
    seg_id_b = 1
    seg_id_cls = 0
    seg_id_pad = 0

    tokens = []
    segment_ids = []

    # [CLS]
    tokens.append("[CLS]")
    segment_ids.append(seg_id_cls)

    # Tokens A
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(seg_id_a)

    # [SEP]
    tokens.append("[SEP]")
    segment_ids.append(seg_id_a)

    # Tokens B (if present)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(seg_id_b)
        tokens.append("[SEP]")
        segment_ids.append(seg_id_b)

    # Convert tokens to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Mask: 1 for real tokens, 0 for padding
    input_mask = [1] * len(input_ids)

    # Padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(seg_id_pad)

    # Sanity checks
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, tokens_a

