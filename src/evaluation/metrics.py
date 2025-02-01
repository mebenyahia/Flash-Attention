import sacrebleu

def compute_bleu(references, hypotheses):
    """
    Computes sacreBLEU score.
    references: list of list of reference sequences (tokens), shape: [N][1..M][tokens]
    hypotheses: list of hypothesis sequences (tokens), shape: [N][tokens]
    """
    ref_texts = []
    for ref_group in references:
        # ref_group might be multiple references, but we only have 1 per sample
        ref_texts.append([" ".join(ref_group[0])])

    hyp_texts = [" ".join(hyp) for hyp in hypotheses]

    # Convert for sacrebleu
    # references for sacrebleu should be list of reference sets => transpose
    # but we only have single reference
    ref_texts_single = [x[0] for x in ref_texts]
    bleu = sacrebleu.corpus_bleu(hyp_texts, [ref_texts_single])
    return bleu.score
