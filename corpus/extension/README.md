# Expanded experiment: grammar and opposites

`grammar_and_contrasts.txt` contains 904 new synthetic teaching passages generated
by `build_extension.py`: 424 grammar passages and 480 contextual contrasts.
Grammar examples vary animals, singular/plural subjects, pronouns, destinations,
agreement, and present/past/progressive forms. Contrasts vary adjectives, objects,
locations, descriptions, comparisons, and transitions in both directions.

The sources are original synthetic teaching material, not private documents or
copied evaluation stories. The generator does not read the evaluation suite,
answer key, model outputs, or chat logs. Shared language facts and vocabulary can
overlap the public benchmark; its exact prompts and answer lists are not training
material. The notebook rejects normalized contiguous prompt matches.

Contextual contrasts do not necessarily teach the question word "opposite" or
transfer to an unfamiliar instruction. Failures are retained and discussed.
This README is ignored by the corpus loader. Only the TXT is imported.
